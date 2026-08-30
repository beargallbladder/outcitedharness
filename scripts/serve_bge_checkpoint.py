#!/usr/bin/env python3
"""Serve dense BGE embeddings from an immutable local checkpoint on loopback."""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def validate_embed_request(
    payload: Any,
    *,
    maximum_batch: int,
) -> tuple[list[str], int]:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    texts = payload.get("texts")
    if (
        not isinstance(texts, list)
        or not texts
        or len(texts) > maximum_batch
        or any(not isinstance(text, str) or not text.strip() for text in texts)
    ):
        raise ValueError(
            f"texts must contain 1..{maximum_batch} non-empty strings"
        )
    requested = payload.get("batch_size", maximum_batch)
    if not isinstance(requested, int) or not 1 <= requested <= maximum_batch:
        raise ValueError(f"batch_size must be between 1 and {maximum_batch}")
    return texts, min(requested, len(texts))


class DenseBGEEncoder:
    def __init__(
        self,
        model_path: Path,
        *,
        device: str,
        maximum_length: int,
        maximum_characters: int,
        dtype: str,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.maximum_length = maximum_length
        self.maximum_characters = maximum_characters
        self._lock = threading.Lock()
        dtypes = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        model_dtype = dtypes[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=model_dtype,
        ).to(device)
        self.model.eval()

    def encode(self, texts: list[str], *, batch_size: int) -> list[list[float]]:
        vectors: list[list[float]] = []
        torch = self.torch
        for start in range(0, len(texts), batch_size):
            chunk = [
                text[: self.maximum_characters]
                for text in texts[start : start + batch_size]
            ]
            tokens = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.maximum_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            with self._lock, torch.inference_mode():
                hidden = self.model(**tokens).last_hidden_state
                dense = torch.nn.functional.normalize(
                    hidden[:, 0].float(),
                    p=2,
                    dim=1,
                )
            vectors.extend(dense.cpu().tolist())
        return vectors


class FlagEmbeddingEncoder:
    """Exact adapter used by the production bge_m3_service.py contract."""

    def __init__(self, model_path: Path, *, device: str, use_fp16: bool) -> None:
        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel(
            str(model_path),
            use_fp16=use_fp16,
            device=device,
        )
        self._lock = threading.Lock()

    def encode(self, texts: list[str], *, batch_size: int) -> list[list[float]]:
        with self._lock:
            output = self.model.encode(
                [text[:512] for text in texts],
                batch_size=batch_size,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        return [vector.tolist() for vector in output["dense_vecs"]]


def make_handler(encoder: Any, *, maximum_batch: int):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HarnessOfflineBGE/1"

        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_error(404)
                return
            self._json(200, {"status": "ok"})

        def do_POST(self) -> None:
            if self.path != "/embed":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 2 * 1024 * 1024:
                    raise ValueError("request body size is invalid")
                payload = json.loads(self.rfile.read(length))
                texts, batch_size = validate_embed_request(
                    payload,
                    maximum_batch=maximum_batch,
                )
                vectors = encoder.encode(texts, batch_size=batch_size)
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})
                return
            except Exception as exc:
                self._json(500, {"error": type(exc).__name__})
                return
            self._json(200, {"embeddings": vectors})

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} {format % args}", flush=True)

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18881)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backend",
        choices=("transformers", "flagembedding"),
        default="transformers",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-characters", type=int, default=512)
    parser.add_argument("--max-batch", type=int, default=256)
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32", "bfloat16"),
        default="float16",
    )
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1"}:
        parser.error("offline checkpoint server must bind to loopback")
    if not args.model.is_dir():
        parser.error("model checkpoint directory does not exist")
    if not 1024 <= args.port <= 65535:
        parser.error("port is invalid")
    if (
        not 1 <= args.max_length <= 8192
        or not 1 <= args.max_characters <= 100_000
        or not 1 <= args.max_batch <= 1024
    ):
        parser.error("encoder limits are invalid")
    return args


def main() -> int:
    args = parse_args()
    if args.backend == "flagembedding":
        encoder = FlagEmbeddingEncoder(
            args.model,
            device=args.device,
            use_fp16=args.dtype == "float16",
        )
    else:
        encoder = DenseBGEEncoder(
            args.model,
            device=args.device,
            maximum_length=args.max_length,
            maximum_characters=args.max_characters,
            dtype=args.dtype,
        )
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(encoder, maximum_batch=args.max_batch),
    )
    print(
        json.dumps(
            {
                "ready": True,
                "model": str(args.model.resolve()),
                "endpoint": f"http://{args.host}:{args.port}/embed",
                "pooling": "cls",
                "backend": args.backend,
                "dtype": args.dtype,
                "max_length": args.max_length,
                "max_characters": args.max_characters,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
