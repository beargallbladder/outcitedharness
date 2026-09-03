#!/usr/bin/env python3
"""File-backed frozen Qwen3.8 PLE lookup for four-rank native TP."""

from __future__ import annotations

import gc
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn


PLE_KEY = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$"
)


@dataclass(frozen=True)
class TensorLocation:
    path: Path
    byte_offset: int
    rows: int
    columns: int


def _safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"truncated safetensors prefix: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length <= 0 or header_length > 100 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header length: {path}")
        raw_header = handle.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError(f"truncated safetensors header: {path}")
    header = json.loads(raw_header)
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors header object: {path}")
    return 8 + header_length, header


def locate_ple_shards(model_path: Path) -> list[TensorLocation]:
    locations: dict[int, TensorLocation] = {}
    for path in sorted(model_path.glob("model-*.safetensors")):
        data_start, header = _safetensors_header(path)
        for key, value in header.items():
            match = PLE_KEY.search(key)
            if match is None:
                continue
            if (
                not isinstance(value, dict)
                or value.get("dtype") != "BF16"
                or not isinstance(value.get("shape"), list)
                or len(value["shape"]) != 2
                or not isinstance(value.get("data_offsets"), list)
                or len(value["data_offsets"]) != 2
            ):
                raise ValueError(f"invalid PLE tensor metadata: {key}")
            index = int(match.group(1))
            if index in locations:
                raise ValueError(f"duplicate PLE checkpoint shard: {index}")
            rows, columns = (int(item) for item in value["shape"])
            start, end = (int(item) for item in value["data_offsets"])
            if rows <= 0 or columns <= 0 or end - start != rows * columns * 2:
                raise ValueError(f"invalid PLE tensor extent: {key}")
            locations[index] = TensorLocation(
                path=path,
                byte_offset=data_start + start,
                rows=rows,
                columns=columns,
            )
    if sorted(locations) != list(range(128)):
        raise ValueError("Qwen3.8 checkpoint must contain PLE shards 0 through 127")
    ordered = [locations[index] for index in range(128)]
    shapes = {(item.rows, item.columns) for item in ordered}
    if len(shapes) != 1:
        raise ValueError("Qwen3.8 PLE checkpoint shards have inconsistent shapes")
    return ordered


class FileBackedShardedEmbedding(nn.Module):
    """Read selected frozen BF16 rows and gather four column shards."""

    def __init__(
        self,
        *,
        locations: list[TensorLocation],
        device: torch.device,
        process_group: dist.ProcessGroup | None = None,
    ):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("file-backed PLE requires an initialized process group")
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.rank = dist.get_rank(process_group)
        if self.world_size != 4:
            raise ValueError("file-backed PLE qualification requires four ranks")
        self.rows_per_checkpoint_shard = locations[0].rows
        self.embedding_dim = locations[0].columns
        self.num_embeddings = sum(item.rows for item in locations)
        if self.embedding_dim % self.world_size:
            raise ValueError("PLE width is not divisible by the TP world size")
        self.local_width = self.embedding_dim // self.world_size
        self.column_start = self.rank * self.local_width
        self.column_end = self.column_start + self.local_width
        self._locations = locations
        self._maps = [
            np.memmap(
                item.path,
                mode="r",
                dtype=np.dtype("<u2"),
                offset=item.byte_offset,
                shape=(item.rows, item.columns),
                order="C",
            )
            for item in locations
        ]
        self.register_buffer(
            "_device_anchor",
            torch.empty(0, dtype=torch.bfloat16, device=device),
            persistent=False,
        )

    @property
    def weight(self) -> torch.Tensor:
        # Qwen4ExpTextNGramEmbedding consults only weight.device before lookup.
        return self._device_anchor

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        original_shape = tuple(input_ids.shape)
        flat_ids = (
            input_ids.detach()
            .reshape(-1)
            .to(device="cpu", dtype=torch.int64)
            .numpy()
        )
        if flat_ids.size == 0:
            raise ValueError("PLE lookup cannot be empty")
        if int(flat_ids.min()) < 0 or int(flat_ids.max()) >= self.num_embeddings:
            raise IndexError("PLE lookup index is outside the checkpoint table")
        shard_ids = flat_ids // self.rows_per_checkpoint_shard
        row_ids = flat_ids % self.rows_per_checkpoint_shard
        output_bits = np.empty(
            (flat_ids.size, self.local_width),
            dtype=np.uint16,
        )
        for shard_index in np.unique(shard_ids):
            positions = np.flatnonzero(shard_ids == shard_index)
            output_bits[positions] = self._maps[int(shard_index)][
                row_ids[positions],
                self.column_start : self.column_end,
            ]
        local = (
            torch.from_numpy(output_bits)
            .view(torch.bfloat16)
            .reshape(*original_shape, self.local_width)
            .to(self._device_anchor.device)
        )
        pieces = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(pieces, local, group=self.process_group)
        return torch.cat(pieces, dim=-1)


def replace_ple_embedding(
    model: nn.Module,
    *,
    model_path: Path,
) -> tuple[FileBackedShardedEmbedding, dict[str, Any]]:
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith("ple.ple_embedding.ngram_embedding")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one Qwen3.8 PLE embedding module, observed {len(matches)}"
        )
    name, resident = matches[0]
    del matches
    if not isinstance(resident, nn.Embedding):
        raise TypeError(f"resident PLE module has unexpected type: {type(resident)}")
    global_shape = tuple(resident.weight.shape)
    local_weight = (
        resident.weight.to_local()
        if hasattr(resident.weight, "to_local")
        else resident.weight
    )
    local_shape = tuple(local_weight.shape)
    if global_shape != (320_001_536, 160) or local_shape != (320_001_536, 40):
        raise RuntimeError(
            f"resident PLE has unexpected TP geometry: {global_shape} / {local_shape}"
        )
    allocated_before = torch.cuda.memory_allocated()
    parent_name, child_name = name.rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    replacement = FileBackedShardedEmbedding(
        locations=locate_ple_shards(model_path),
        device=resident.weight.device,
    )
    setattr(parent, child_name, replacement)
    del resident
    del local_weight
    gc.collect()
    torch.cuda.empty_cache()
    allocated_after = torch.cuda.memory_allocated()
    recovered = allocated_before - allocated_after
    if recovered < 20 * 1024**3:
        raise RuntimeError(
            f"file-backed PLE recovered only {recovered / 1024**3:.3f} GiB"
        )
    return replacement, {
        "module": name,
        "resident_global_shape": list(global_shape),
        "resident_local_shape": list(local_shape),
        "allocated_before_gib": round(allocated_before / 1024**3, 3),
        "allocated_after_gib": round(allocated_after / 1024**3, 3),
        "recovered_gib": round(recovered / 1024**3, 3),
    }
