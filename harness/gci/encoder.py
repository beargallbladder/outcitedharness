from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from harness.gci.slicing import MAX_SLICE_CHARS


PRODUCTION_ENCODER_URL = "http://127.0.0.1:8800/v1/embeddings"
MAX_ENCODER_BATCH = 16
EMBED_DIM = 1024


class EncoderPolicyError(RuntimeError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise EncoderPolicyError(f"encoder redirect refused: HTTP {code}")


def validate_encoder_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.port != 8800
        or parsed.path != "/v1/embeddings"
        or parsed.query
        or parsed.fragment
    ):
        raise EncoderPolicyError(
            "GCI encoder must be exactly http://127.0.0.1:8800/v1/embeddings"
        )
    return PRODUCTION_ENCODER_URL


@dataclass(frozen=True)
class EncoderResponse:
    vectors: tuple[tuple[float, ...], ...]
    latency_ms: float


class StrictEncoder:
    def __init__(
        self,
        url: str = PRODUCTION_ENCODER_URL,
        *,
        timeout: float = 30.0,
        model: str = "bge-m3-cr-tapes-v1",
    ):
        self.url = validate_encoder_url(url)
        self.timeout = timeout
        self.model = model
        self._opener = urllib.request.build_opener(_NoRedirect())

    def embed(self, texts: list[str]) -> EncoderResponse:
        if not texts:
            return EncoderResponse((), 0.0)
        if len(texts) > MAX_ENCODER_BATCH:
            raise EncoderPolicyError(
                f"encoder batch {len(texts)} exceeds maximum {MAX_ENCODER_BATCH}"
            )
        if any(len(text) > MAX_SLICE_CHARS for text in texts):
            raise EncoderPolicyError(
                f"encoder input exceeds {MAX_SLICE_CHARS}-character GCI contract"
            )
        request = urllib.request.Request(
            self.url,
            data=json.dumps({"model": self.model, "input": texts}).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8") or "{}")
        except EncoderPolicyError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"encoder request failed: {type(exc).__name__}: {exc}") from exc
        latency = (time.perf_counter() - started) * 1000
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise RuntimeError("encoder returned unexpected row count")
        ordered = sorted(rows, key=lambda row: int(row.get("index", 0)))
        vectors = []
        for row in ordered:
            vector = row.get("embedding") if isinstance(row, dict) else None
            if not isinstance(vector, list) or len(vector) != EMBED_DIM:
                raise RuntimeError(f"encoder returned non-{EMBED_DIM}-dimensional vector")
            vectors.append(tuple(float(value) for value in vector))
        return EncoderResponse(tuple(vectors), latency)
