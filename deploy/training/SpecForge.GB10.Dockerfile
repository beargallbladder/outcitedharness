FROM nvcr.io/nvidia/pytorch@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3

ARG SPECFORGE_COMMIT=c439546983863facd8126f505c2d291d0ab31faf

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN git init /opt/specforge \
    && git -C /opt/specforge remote add origin https://github.com/sgl-project/SpecForge.git \
    && git -C /opt/specforge fetch --depth 1 origin "${SPECFORGE_COMMIT}" \
    && git -C /opt/specforge checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/specforge rev-parse HEAD)" = "${SPECFORGE_COMMIT}"

RUN python -m pip install --no-cache-dir \
      "transformers==5.12.1" \
      datasets \
      tqdm \
      accelerate \
      huggingface-hub \
      numpy \
      openai-harmony \
      pydantic \
      psutil \
      pyyaml \
      safetensors \
      "sglang==0.5.18" \
      requests \
      tensorboard \
      typing-extensions \
      wandb \
      yunchang \
    && python -m pip install --no-cache-dir --no-deps /opt/specforge

RUN python -c "import platform, torch, transformers; assert platform.machine() == 'aarch64'; assert torch.__version__.startswith('2.13.0'); assert transformers.__version__ == '5.12.1'" \
    && specforge --help >/dev/null

WORKDIR /opt/specforge
ENV PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENTRYPOINT ["specforge"]
