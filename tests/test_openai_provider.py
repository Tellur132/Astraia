"""Unit tests for the OpenAI provider shim."""

from types import SimpleNamespace

import pytest

from astraia.llm_providers import OpenAIProvider, Prompt, PromptMessage, ToolDefinition
import astraia.llm_providers.openai as openai_module


class _DummyResponses:
    def __init__(self, sink: dict):
        self._sink = sink

    def create(self, **kwargs):  # pragma: no cover - exercised via provider
        self._sink["kwargs"] = kwargs
        return SimpleNamespace(output_text="{}", usage=None)


class _DummyClient:
    def __init__(self, sink: dict, **_kwargs):
        self.responses = _DummyResponses(sink)


@pytest.fixture()
def monkeypatched_client(monkeypatch):
    sink: dict = {}

    def _factory(**kwargs):
        return _DummyClient(sink, **kwargs)

    monkeypatch.setattr(openai_module, "OpenAI", _factory)
    return sink


def test_openai_provider_emits_responses_tool_payload(monkeypatched_client):
    provider = OpenAIProvider(model="gpt-test", api_key="fake")
    prompt = Prompt(messages=[PromptMessage(role="user", content="ping")])
    tool = ToolDefinition(
        name="plan_next_batch",
        description="Return a JSON strategy",
        parameters={"type": "object", "properties": {}},
    )

    provider.generate(prompt, tool=tool)

    kwargs = monkeypatched_client["kwargs"]
    assert kwargs["tools"] == [
        {
            "type": "function",
            "name": tool.name,
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
    ]
    assert kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": tool.name},
    }
