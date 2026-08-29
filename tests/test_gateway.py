from pathlib import Path

import pytest
from starlette.requests import Request

from harness.gateway.qwen_tools import parse_qwen_tool_text, rewrite_openai_completion
from harness.gateway.server import _authorized
from harness.gateway.spec import GatewaySpec, listed_models, resolve_alias


def _spec() -> GatewaySpec:
    return GatewaySpec(
        listen_host="127.0.0.1",
        listen_port=8787,
        api_key="harness-local",
        aliases={
            "harness-auto": "auto",
            "harness-local": "dgx_qwen",
            "harness-m5": "m5_qwen",
            "harness-frontier": "frontier",
        },
        auto_ladder=["dgx_qwen", "m5_qwen", "frontier"],
        context_window=131072,
        max_output_tokens=8192,
    )


def _request(headers: list[tuple[bytes, bytes]], host: str = "127.0.0.1") -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": headers,
            "client": (host, 12345),
        }
    )


def test_loopback_accepts_any_client_key():
    spec = _spec()
    req = _request([(b"authorization", b"Bearer leftover-openrouter-key")])
    assert _authorized(req, spec)
    assert _authorized(_request([]), spec)
    assert _authorized(_request([(b"api-key", b"sk-harness-local")]), spec)


def test_non_loopback_requires_configured_key():
    spec = GatewaySpec(
        listen_host="0.0.0.0",
        listen_port=8787,
        api_key="harness-local",
        aliases={},
        auto_ladder=[],
        context_window=131072,
        max_output_tokens=8192,
    )
    remote = "203.0.113.9"
    assert _authorized(_request([(b"authorization", b"Bearer harness-local")], remote), spec)
    assert _authorized(_request([(b"api-key", b"sk-harness-local")], remote), spec)
    assert not _authorized(_request([(b"authorization", b"Bearer other")], remote), spec)


def test_alias_maps_gateway_ids():
    spec = _spec()
    assert resolve_alias(spec, "harness-local") == "dgx_qwen"
    assert resolve_alias(spec, "harness-auto") == "auto"
    assert resolve_alias(spec, "dgx_qwen") == "dgx_qwen"


def test_models_list_exposes_gateway_ids():
    ids = {row["id"] for row in listed_models(_spec())}
    assert ids == {"harness-auto", "harness-local", "harness-m5", "harness-frontier"}


def test_remote_gateway_exposes_only_harness_identity(tmp_path: Path):
    from harness.config import AppConfig, Settings
    from harness.gateway.server import create_app
    from starlette.testclient import TestClient

    cfg = AppConfig(
        root=tmp_path,
        settings=Settings(results_dir=tmp_path / "results", db_path=tmp_path / "h.db"),
        models={},
        pricing={},
    )
    spec = _spec()
    spec.listen_host = "0.0.0.0"
    spec.aliases["harness-orch"] = "orch"
    client = TestClient(create_app(cfg, spec))

    models = client.get("/v1/models").json()["data"]
    assert [row["id"] for row in models] == ["harness-orch"]
    health = client.get("/healthz").json()
    assert health == {"ready": True, "service": "harness", "model": "harness-orch"}
    index = client.get("/").json()
    assert index["models"] == ["harness-orch"]
    assert spec.api_key not in str(index)
    unauth = client.post(
        "/v1/chat/completions",
        json={"model": "harness-local", "messages": []},
    )
    assert unauth.status_code == 401


def test_authenticated_remote_still_gets_only_orchestration_model(tmp_path: Path):
    from harness.config import AppConfig, Settings
    from harness.gateway.server import create_app
    from starlette.testclient import TestClient

    cfg = AppConfig(
        root=tmp_path,
        settings=Settings(results_dir=tmp_path / "results", db_path=tmp_path / "h.db"),
        models={},
        pricing={},
    )
    spec = _spec()
    spec.listen_host = "0.0.0.0"
    spec.aliases["harness-orch"] = "orch"
    client = TestClient(create_app(cfg, spec))
    auth = {"Authorization": f"Bearer {spec.api_key}"}

    ids = {row["id"] for row in client.get("/v1/models", headers=auth).json()["data"]}
    assert ids == {"harness-orch"}
    health = client.get("/healthz", headers=auth).json()
    assert health["service"] == "harness-orch-gateway"
    index = client.get("/", headers=auth).json()
    assert index["models"] == ["harness-orch"]
    rejected = client.post(
        "/v1/chat/completions",
        headers=auth,
        json={"model": "harness-local", "messages": []},
    )
    assert rejected.status_code == 404
    assert rejected.json()["error"]["message"] == "unknown harness model"


def test_qwen_xml_becomes_openai_tool_calls():
    text = """<tool_call>
<function=execute_command>
<parameter=command>
git clone https://github.com/octocat/Hello-World.git /tmp/hello-world
</parameter>
</function>
</tool_call>"""
    calls = parse_qwen_tool_text(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "execute_command"
    assert "git clone" in calls[0]["function"]["arguments"]
    rewritten = rewrite_openai_completion(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    assert rewritten["choices"][0]["finish_reason"] == "tool_calls"
    assert rewritten["choices"][0]["message"]["content"] == ""
    assert rewritten["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "execute_command"


def test_last_user_text_and_orch_alias():
    from harness.gateway.orch import last_user_text
    from harness.gateway.spec import is_orch_alias

    spec = _spec()
    spec.aliases["harness-orch"] = "orch"
    assert is_orch_alias(spec, "harness-orch") is True
    assert is_orch_alias(spec, "harness-local") is False
    assert last_user_text([{"role": "system", "content": "x"}, {"role": "user", "content": "split this"}]) == "split this"
    from harness.gateway.orch import compact_thread

    thread = compact_thread(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "split this"},
        ]
    )
    assert "first" in thread
    assert "split this" in thread


def test_client_workspace_root_from_env_block(tmp_path: Path):
    from harness.gateway.orch import client_workspace_root

    root = tmp_path / "locationlocationlocation"
    root.mkdir()
    messages = [
        {
            "role": "system",
            "content": (
                "You are Cursor.\n"
                "<env>\n"
                "1. Platform: darwin\n"
                "3. IDE: VS Code\n"
                f"4. Working Directory: {root}/\n"
                "</env>\n"
            ),
        },
        {"role": "user", "content": "review the score"},
    ]
    assert client_workspace_root(messages) == root.resolve()


def test_client_workspace_root_from_current_system_information(tmp_path: Path):
    from harness.gateway.orch import client_workspace_root

    root = tmp_path / "greenfield"
    root.mkdir()
    messages = [
        {
            "role": "system",
            "content": (
                "You are Cursor.\n\n"
                "SYSTEM INFORMATION\n"
                "Operating System: macOS\n"
                "IDE: Visual Studio Code\n"
                f"Primary Working Directory: {root}\n"
            ),
        },
        {"role": "user", "content": "Continue greenfield gf-example"},
    ]
    assert client_workspace_root(messages) == root.resolve()


def test_client_workspace_root_disagreement_fails_closed(tmp_path: Path):
    from harness.gateway.orch import client_workspace_root

    a = tmp_path / "repoA"
    b = tmp_path / "repoB"
    a.mkdir()
    b.mkdir()
    messages = [
        {
            "role": "system",
            "content": f"<env>\n4. Working Directory: {a}\n</env>",
        }
    ]
    extra = {"cwd": str(b)}
    assert client_workspace_root(messages, extra) is None
    assert client_workspace_root([{"role": "user", "content": "no env here"}]) is None
    assert client_workspace_root(messages, extra={"cwd": "relative/path"}) == a.resolve()


def test_orch_complete_stitches(tmp_path: Path, monkeypatch):
    from harness.config import AppConfig, Settings
    from harness.gateway.orch import OrchResult
    from harness.gateway.server import create_app
    from starlette.testclient import TestClient

    async def fake_run(cfg, intent, thread="", messages=None, tools=None, extra=None):
        return OrchResult(text=f"STITCHED:{intent}")

    monkeypatch.setattr("harness.gateway.orch.run_orch", fake_run)
    cfg = AppConfig(
        root=tmp_path,
        settings=Settings(results_dir=tmp_path / "results", db_path=tmp_path / "h.db"),
        models={},
        pricing={},
    )
    spec = _spec()
    spec.aliases["harness-orch"] = "orch"
    client = TestClient(create_app(cfg, spec))
    body = client.post(
        "/v1/chat/completions",
        json={
            "model": "harness-orch",
            "messages": [{"role": "user", "content": "read ARCHITECTURE.md"}],
            "stream": False,
        },
    ).json()
    assert "STITCHED:read ARCHITECTURE.md" in body["choices"][0]["message"]["content"]


def test_orch_gather_returns_tool_calls(tmp_path: Path, monkeypatch):
    from harness.config import AppConfig, Settings
    from harness.gateway.orch import OrchResult
    from harness.gateway.server import create_app
    from starlette.testclient import TestClient

    async def fake_run(cfg, intent, thread="", messages=None, tools=None, extra=None):
        return OrchResult(
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"ARCHITECTURE.md"}'},
                }
            ]
        )

    monkeypatch.setattr("harness.gateway.orch.run_orch", fake_run)
    cfg = AppConfig(
        root=tmp_path,
        settings=Settings(results_dir=tmp_path / "results", db_path=tmp_path / "h.db"),
        models={},
        pricing={},
    )
    spec = _spec()
    spec.aliases["harness-orch"] = "orch"
    client = TestClient(create_app(cfg, spec))
    body = client.post(
        "/v1/chat/completions",
        json={
            "model": "harness-orch",
            "messages": [{"role": "user", "content": "review this repo"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                    },
                }
            ],
            "stream": False,
        },
    ).json()
    msg = body["choices"][0]["message"]
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert msg["tool_calls"][0]["function"]["name"] == "read_file"
    assert msg["content"] == ""


def test_compact_thread_keeps_tool_results():
    from harness.gateway.orch import compact_thread, last_user_text

    messages = [
        {"role": "user", "content": "review this project"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"package.json"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"name":"outcited"}'},
    ]
    thread = compact_thread(messages)
    assert "package.json" in thread
    assert "outcited" in thread
    assert last_user_text(messages) == "review this project"


@pytest.mark.asyncio
async def test_no_repo_evidence_gathers_immediately(monkeypatch):
    from harness.gateway.orch import has_repo_evidence, run_orch

    assert has_repo_evidence([{"role": "user", "content": "review this"}]) is False
    assert has_repo_evidence(
        [
            {"role": "assistant", "content": "Harness orch x\nQA FAIL closed: nope"},
            {"role": "user", "content": "review this"},
        ]
    ) is False
    assert has_repo_evidence(
        [{"role": "tool", "tool_call_id": "call_1", "content": '{"name":"outcited","reach":120}'}]
    ) is True

    async def boom(*args, **kwargs):
        raise AssertionError("dispatch must not run before the client gathers")

    monkeypatch.setattr("harness.gateway.orch.run_dispatch", boom)
    monkeypatch.setattr("harness.gateway.orch.plan_orch", boom)

    class Cfg:
        models = {}

    result = await run_orch(
        Cfg(),
        "verify category rank consensus math",
        messages=[{"role": "user", "content": "verify category rank consensus math"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "parameters": {"type": "object", "properties": {"path": {}, "regex": {}}},
                },
            }
        ],
    )
    assert result.tool_calls
    assert result.tool_calls[0]["function"]["name"] in {"search_files", "list_files", "execute_command", "read_file"}


@pytest.mark.asyncio
async def test_two_evidence_gathers_force_dispatch(monkeypatch):
    from harness.dispatch import DispatchReport
    from harness.gateway.orch import run_orch

    messages = [{"role": "user", "content": "review this code base"}]
    for index in range(2):
        call_id = f"call_{index}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "read_files", "arguments": '{"paths":["src/app.py"]}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": "src/app.py\n" + ("def work(): pass\n" * 20),
                },
            ]
        )

    called = {}

    async def pick(*args, **kwargs):
        return "m5_qwen", object()

    async def no_more_planning(*args, **kwargs):
        raise AssertionError("two successful gather rounds must not ask for more")

    async def dispatch(_cfg, intent, **kwargs):
        called["thread"] = kwargs["thread"]
        return DispatchReport(run_id="forced", intent=intent, slice_error="test stop")

    monkeypatch.setattr("harness.gateway.orch.pick_foreman", pick)
    monkeypatch.setattr("harness.gateway.orch.plan_orch", no_more_planning)
    monkeypatch.setattr("harness.gateway.orch.run_dispatch", dispatch)

    result = await run_orch(
        object(),
        "review this code base",
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_files",
                    "parameters": {"type": "object", "properties": {"paths": {}}},
                },
            }
        ],
    )
    assert "src/app.py" in called["thread"]
    assert "test stop" in result.text


@pytest.mark.asyncio
async def test_frontend_thin_evidence_keeps_gathering(monkeypatch):
    from harness.gateway.orch import run_orch

    messages = [{"role": "user", "content": "how about the front end. is it clear, engaging?"}]
    for index in range(2):
        call_id = f"call_{index}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "read_files", "arguments": '{"paths":["README.md"]}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": "- `apps/web` — thin debug / CRE surface\n{\"name\":\"@locdna/web\"}",
                },
            ]
        )

    async def boom(*args, **kwargs):
        raise AssertionError("thin frontend evidence must not dispatch a no-frontend packet")

    monkeypatch.setattr("harness.gateway.orch.pick_foreman", boom)
    monkeypatch.setattr("harness.gateway.orch.plan_orch", boom)
    monkeypatch.setattr("harness.gateway.orch.run_dispatch", boom)

    result = await run_orch(
        object(),
        "how about the front end. is it clear, engaging?",
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_files",
                    "parameters": {"type": "object", "properties": {"paths": {}}},
                },
            }
        ],
    )
    assert result.tool_calls
    blob = " ".join(call["function"]["arguments"] for call in result.tool_calls)
    assert "apps/web" in blob


@pytest.mark.asyncio
async def test_named_source_listing_keeps_gathering(monkeypatch):
    from harness.gateway.orch import run_orch

    intent = "fix the failing unit test in test_add.py"
    messages = [{"role": "user", "content": intent}]
    for index in range(2):
        call_id = f"call_{index}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "list_files",
                                "arguments": '{"path":".","recursive":false}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": ".git/\nadd.py\ntest_add.py\nREADME.md\n",
                },
            ]
        )

    async def boom(*args, **kwargs):
        raise AssertionError("listing named source files must not dispatch")

    monkeypatch.setattr("harness.gateway.orch.pick_foreman", boom)
    monkeypatch.setattr("harness.gateway.orch.plan_orch", boom)
    monkeypatch.setattr("harness.gateway.orch.run_dispatch", boom)

    result = await run_orch(
        object(),
        intent,
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_files",
                    "parameters": {"type": "object", "properties": {"paths": {}}},
                },
            }
        ],
    )
    assert result.tool_calls
    blob = " ".join(call["function"]["arguments"] for call in result.tool_calls)
    assert "test_add.py" in blob
    assert "add.py" in blob
