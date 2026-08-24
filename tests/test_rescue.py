from pathlib import Path

import pytest

from harness.rescue import PACKET_TEMPLATE, PacketError, load_packet, missing_sections


def test_template_has_every_section():
    assert missing_sections(PACKET_TEMPLATE) == []


def test_load_packet_rejects_short_dump(tmp_path: Path):
    p = tmp_path / "bad.md"
    p.write_text("# TASK\nclone something\n")
    with pytest.raises(PacketError, match="missing sections"):
        load_packet(p)


def test_load_packet_rejects_cline_dump(tmp_path: Path):
    p = tmp_path / "huge.md"
    p.write_text(PACKET_TEMPLATE + ("x" * 20_001))
    with pytest.raises(PacketError, match="20k"):
        load_packet(p)


def test_load_packet_accepts_filled_template(tmp_path: Path):
    p = tmp_path / "ok.md"
    p.write_text(PACKET_TEMPLATE.replace("# TASK", "# TASK\nfix the pipefail script"))
    assert "pipefail" in load_packet(p)
