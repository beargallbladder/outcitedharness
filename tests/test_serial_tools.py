from pathlib import Path

from harness.serial.tools import parse_tool, resolve_in, run_tool


def test_parse_tool():
    call = parse_tool("<tool name=\"read\"><path>lib/api/v1/auth.ts</path></tool>")
    assert call is not None
    assert call.name == "read"
    assert call.args["path"] == "lib/api/v1/auth.ts"


def test_path_jail(tmp_path: Path):
    try:
        resolve_in(tmp_path, "../secret")
    except ValueError:
        return
    raise AssertionError("escaped")


def test_strreplace_and_run(tmp_path: Path):
    (tmp_path / "f.txt").write_text("custom return true\n")
    oracle = tmp_path / "check.py"
    oracle.write_text("import pathlib,sys\n"
                      "t=pathlib.Path('f.txt').read_text()\n"
                      "sys.exit(0 if 'fixed' in t else 1)\n")
    from harness.serial.tools import ToolCall

    out = run_tool(
        tmp_path,
        oracle,
        ToolCall("strreplace", {"path": "f.txt", "old": "custom return true", "new": "fixed"}),
    )
    assert out.startswith("OK")
    check = run_tool(tmp_path, oracle, ToolCall("run", {}))
    assert check.startswith("PASS")
