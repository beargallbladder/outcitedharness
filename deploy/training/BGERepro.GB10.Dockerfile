FROM harness/llamafactory-gb10:20260829

RUN python -m venv /opt/bge-repro

ENV PATH="/opt/bge-repro/bin:${PATH}"

RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu130 \
    "torch==2.11.0"

RUN python -m pip install --no-cache-dir \
    "transformers==4.57.6" \
    "numpy==2.4.4" \
    "tokenizers==0.22.2" \
    "safetensors==0.7.0" \
    "accelerate==1.13.0" \
    "huggingface-hub==0.36.2" \
    "sentencepiece==0.2.1" \
    "scipy==1.17.1" \
    && python -m pip install --no-cache-dir "FlagEmbedding==1.4.0"
