from starlette.requests import Request

from harness.gateway.anthropic_compat import openai_to_anthropic_payload
from harness.gateway.qwen_tools import parse_qwen_tool_text, rewrite_openai_completion
from harness.gateway.server import _authorized
from harness.gateway.spec import ClineSpec, listed_models, resolve_alias


def _spec() -> ClineSpec:
    return ClineSpec(
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


def test_loopback_accepts_any_cline_key():
    spec = _spec()
    req = _request([(b"authorization", b"Bearer leftover-openrouter-key")])
    assert _authorized(req, spec)
    assert _authorized(_request([]), spec)
    assert _authorized(_request([(b"api-key", b"sk-harness-local")]), spec)


def test_non_loopback_requires_configured_key():
    spec = ClineSpec(
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


def test_alias_maps_cline_ids():
    spec = _spec()
    assert resolve_alias(spec, "harness-local") == "dgx_qwen"
    assert resolve_alias(spec, "harness-auto") == "auto"
    assert resolve_alias(spec, "dgx_qwen") == "dgx_qwen"


def test_models_list_exposes_cline_ids():
    ids = {row["id"] for row in listed_models(_spec())}
    assert ids == {"harness-auto", "harness-local", "harness-m5", "harness-frontier"}


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


def test_openai_tools_become_anthropic_tools():
    body = {
        "model": "harness-frontier",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "read foo"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ],
        "max_tokens": 99,
    }
    payload = openai_to_anthropic_payload(body, "claude-sonnet-4-6", 8192)
    assert payload["model"] == "claude-sonnet-4-6"
    assert payload["system"] == "Be brief."
    assert payload["max_tokens"] == 99
    assert payload["tools"][0]["name"] == "read_file"
    assert payload["tools"][0]["input_schema"]["properties"]["path"]["type"] == "string"
