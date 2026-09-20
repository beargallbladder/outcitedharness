from __future__ import annotations

import base64
from pathlib import Path

from harness.cases.prompt import build_prompt
from harness.cases.schema import Case, EvaluationSpec
from harness.config import Capabilities, ModelConfig, Settings
from harness.providers.anthropic import _message_blocks
from harness.providers.base import ChatMessage, ImageAttachment
from harness.providers.openai_compatible import _message_payload


def _model(vision: bool) -> ModelConfig:
    return ModelConfig(
        key="test_model",
        tier=0,
        display_name="Test Model",
        short_name="TEST",
        provider="openai_compatible",
        base_url="http://127.0.0.1:9/v1",
        model="test-model",
        capabilities=Capabilities(vision=vision),
    )


def _settings() -> Settings:
    return Settings(results_dir=Path("/tmp/harness-results"), db_path=Path("/tmp/harness.db"))


def _case(tmp_path: Path) -> Case:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "photo.png").write_bytes(b"\x89PNG fake image bytes")
    (inputs / "plot.jpeg").write_bytes(b"\xff\xd8 fake jpeg")
    (inputs / "blob.bin").write_bytes(b"\x00\x01 binary not an image")
    (inputs / "notes.txt").write_text("textual evidence")
    return Case(
        id="vision_case",
        title="Vision case",
        category="vision",
        path=tmp_path,
        prompt="Describe the attached images.",
        input_files=["inputs/photo.png", "inputs/plot.jpeg", "inputs/blob.bin", "inputs/notes.txt"],
        evaluation=EvaluationSpec(type="human"),
    )


def test_vision_model_attaches_images_and_skips_non_images(tmp_path: Path) -> None:
    packet = build_prompt(_case(tmp_path), _settings(), _model(vision=True))
    assert [image.mime_type for image in packet.images] == ["image/png", "image/jpeg"]
    assert packet.images[0].data_b64 == base64.b64encode(b"\x89PNG fake image bytes").decode()
    assert packet.skipped_binaries == ["blob.bin"]
    assert "photo.png" not in packet.user
    assert "blob.bin" in packet.user


def test_non_vision_model_sends_nothing_and_says_so(tmp_path: Path) -> None:
    packet = build_prompt(_case(tmp_path), _settings(), _model(vision=False))
    assert packet.images == []
    assert packet.skipped_binaries == ["photo.png", "plot.jpeg", "blob.bin"]
    assert "configured without native vision" in packet.user


def test_openai_payload_is_multipart_only_with_images() -> None:
    plain = _message_payload(ChatMessage(role="user", content="hi"))
    assert plain == {"role": "user", "content": "hi"}
    rich = _message_payload(
        ChatMessage(
            role="user",
            content="hi",
            images=[ImageAttachment(mime_type="image/png", data_b64="QUJD")],
        )
    )
    assert rich["content"][0] == {"type": "text", "text": "hi"}
    assert rich["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,QUJD"},
    }


def test_anthropic_blocks_use_base64_source() -> None:
    rich = _message_blocks(
        ChatMessage(
            role="user",
            content="hi",
            images=[ImageAttachment(mime_type="image/png", data_b64="QUJD")],
        )
    )
    assert rich["content"][0] == {"type": "text", "text": "hi"}
    assert rich["content"][1] == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "QUJD",
        },
    }
    assert _message_blocks(ChatMessage(role="user", content="hi")) == {
        "role": "user",
        "content": "hi",
    }
