ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.06-py3
FROM ${BASE_IMAGE}

WORKDIR /opt/llamafactory
COPY . .

# LlamaFactory imports TorchAudio at module import time even for text-only jobs.
# NVIDIA's GB10 image intentionally omits it, while PyPI's wheel targets a
# different CUDA minor. Make the import audio-path-local and remove the
# incompatible wheel after dependency resolution.
RUN python - <<'PY'
from pathlib import Path

path = Path("src/llamafactory/data/mm_plugin.py")
text = path.read_text()
text = text.replace("import torchaudio\n", "")
needle = '''        results, sampling_rates = [], []
        for audio in audios:
'''
replacement = '''        import torchaudio

        results, sampling_rates = [], []
        for audio in audios:
'''
if needle not in text:
    raise SystemExit("LlamaFactory audio import patch no longer applies")
path.write_text(text.replace(needle, replacement))
PY

RUN python -m pip install --no-cache-dir -e . \
    && python -m pip uninstall -y torchaudio torchao

ENTRYPOINT ["llamafactory-cli"]
