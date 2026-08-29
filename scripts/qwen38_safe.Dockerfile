FROM oxbyte/qwen3.8-flash-next-dual-spark@sha256:836355c535fea41e5600cf581c69daffd5e27a743192d857e0682876164971b9

# vLLM persistent_topk can silently drop QSA candidates (vllm#51782).
# Use the exact torch path until the upstream CUDA fix is merged and shipped.
RUN python3 - <<'PY'
from pathlib import Path

path = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/"
    "qwen3_8_flash_next/nvidia/ops/qsa.py"
)
source = path.read_text()
old = """\
        topk_op = (
            torch.ops._C.cooperative_topk
            if use_cooperative_topk
            else torch.ops._C.persistent_topk
        )
        topk_op(logits, visible_blocks, blocks, topk_workspace, block_topk, columns)
"""
new = """\
        if use_cooperative_topk:
            torch.ops._C.cooperative_topk(
                logits,
                visible_blocks,
                blocks,
                topk_workspace,
                block_topk,
                columns,
            )
        else:
            # Validated mode-3 fallback from k3net/docai-evals: exact radix
            # selection followed by canonical value/index ordering and padding.
            exact_logits = logits[:, :columns].float()
            column_ids = torch.arange(
                exact_logits.shape[1], device=exact_logits.device
            ).unsqueeze(0)
            lengths = visible_blocks.to(torch.int64).clamp(
                min=0, max=exact_logits.shape[1]
            ).unsqueeze(1)
            exact_logits = exact_logits.masked_fill(
                column_ids >= lengths, float("-inf")
            )
            select_count = min(block_topk, exact_logits.shape[1])
            values, selected = torch.topk(
                exact_logits,
                select_count,
                dim=1,
                largest=True,
                sorted=False,
            )
            by_index = torch.sort(selected, dim=1, stable=True)
            values = values.gather(1, by_index.indices)
            selected = by_index.values
            by_value = torch.sort(values, dim=1, descending=True, stable=True)
            selected = selected.gather(1, by_value.indices)
            canonical = torch.full(
                (exact_logits.shape[0], block_topk),
                -1,
                dtype=torch.int32,
                device=exact_logits.device,
            )
            positions = torch.arange(
                select_count, device=exact_logits.device
            ).unsqueeze(0)
            canonical[:, :select_count] = torch.where(
                positions < lengths,
                selected,
                torch.full_like(selected, -1),
            ).to(torch.int32)
            blocks.copy_(canonical)
"""
if source.count(old) != 1:
    raise RuntimeError(f"expected one QSA top-k block, found {source.count(old)}")
path.write_text(source.replace(old, new, 1))
PY

COPY qwen38_safe_serve.sh /usr/local/bin/qwen38-safe-serve
RUN chmod +x /usr/local/bin/qwen38-safe-serve

ENTRYPOINT ["/usr/local/bin/qwen38-safe-serve"]
