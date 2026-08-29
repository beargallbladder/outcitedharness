ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.06-py3
FROM ${BASE_IMAGE}

WORKDIR /opt/llamafactory
COPY . .

RUN python -m pip install --no-cache-dir -e .

ENTRYPOINT ["llamafactory-cli"]
