ARG BASE_IMAGE=lmsysorg/sglang@sha256:14ed582518584c5c830206b5318a2c2769e68229c3422e48a28b952b3a888bd4
FROM ${BASE_IMAGE}

RUN python3 - <<'PY'
from pathlib import Path

qwen_path = Path(
    "/sgl-workspace/sglang/python/sglang/srt/models/qwen3_next.py"
)
qwen = qwen_path.read_text()
old_qwen = """\
        self._override_weight_loader(
            self.in_proj_qkvz, self._make_packed_weight_loader(self.in_proj_qkvz)
        )
        self._override_weight_loader(
            self.in_proj_ba, self._make_packed_weight_loader(self.in_proj_ba)
        )
"""
new_qwen = """\
        if hasattr(self.in_proj_qkvz, "weight") and hasattr(
            self.in_proj_qkvz.weight, "weight_loader"
        ):
            self._override_weight_loader(
                self.in_proj_qkvz,
                self._make_packed_weight_loader(self.in_proj_qkvz),
            )
        if hasattr(self.in_proj_ba, "weight") and hasattr(
            self.in_proj_ba.weight, "weight_loader"
        ):
            self._override_weight_loader(
                self.in_proj_ba,
                self._make_packed_weight_loader(self.in_proj_ba),
            )
"""
if old_qwen not in qwen:
    raise SystemExit("Qwen3-Next packed-loader patch context not found")
qwen_path.write_text(qwen.replace(old_qwen, new_qwen, 1))

expert_path = Path(
    "/sgl-workspace/sglang/python/sglang/srt/eplb/expert_location.py"
)
expert = expert_path.read_text()
old_expert = """\
        model_class, _ = get_model_architecture(model_config)
        if hasattr(model_class, "get_model_config_for_expert_location"):
"""
new_expert = """\
        try:
            model_class, _ = get_model_architecture(model_config)
        except (ValueError, KeyError):
            return None
        if hasattr(model_class, "get_model_config_for_expert_location"):
"""
if old_expert not in expert:
    raise SystemExit("Expert-location patch context not found")
expert_path.write_text(expert.replace(old_expert, new_expert, 1))
PY
