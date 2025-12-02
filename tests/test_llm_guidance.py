from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from typing import Any, List, Mapping

from astraia.llm_guidance import LLMProposalGenerator, PromptCache
from astraia.llm_providers import LLMExchangeLogger, LLMResult, LLMUsage, LLMUsageLogger


class StubProvider:
    def __init__(self, responses: List[LLMResult]) -> None:
        self._responses = deque(responses)
        self.temperatures: List[float | None] = []

    def generate(
        self,
        prompt,
        *,
        temperature=None,
        json_mode=False,
        system=None,
        tool=None,
    ):  # noqa: ANN001
        self.temperatures.append(temperature)
        if not self._responses:
            raise RuntimeError("No more responses queued")
        return self._responses.popleft()


def make_search_space() -> dict[str, dict[str, Any]]:
    return {
        "theta": {"type": "float", "low": -1.0, "high": 1.0, "step": None, "log": False},
        "depth": {"type": "int", "low": 1, "high": 3, "step": 1, "log": None},
        "backend": {"type": "categorical", "choices": ["a", "b", "c"]},
    }


def make_search_space_with_llm_only() -> dict[str, dict[str, Any]]:
    space = make_search_space()
    space["circuit_code"] = {"type": "llm_only", "default": "<llm>"}
    return space


def make_generator(
    responses: List[LLMResult],
    *,
    tmp_path: Path,
    base_temperature: float = 0.6,
    min_temperature: float = 0.2,
    max_retries: int = 2,
    search_space: Mapping[str, Mapping[str, Any]] | None = None,
    batch_size: int = 2,
    trace_path: Path | None = None,
) -> LLMProposalGenerator:
    provider = StubProvider(responses)
    usage_logger = LLMUsageLogger(tmp_path / "usage.csv")
    trace_logger = LLMExchangeLogger(trace_path) if trace_path is not None else None
    generator = LLMProposalGenerator(
        search_space=search_space or make_search_space(),
        problem_summary="demo",
        objective="minimize",
        batch_size=batch_size,
        base_temperature=base_temperature,
        min_temperature=min_temperature,
        max_retries=max_retries,
        provider=provider,
        usage_logger=usage_logger,
        trace_logger=trace_logger,
        cache=PromptCache(),
        seed=123,
    )
    setattr(generator, "_test_provider", provider)
    return generator


def result_from_payload(payload: Any) -> LLMResult:
    usage = LLMUsage(provider="stub", model="stub", request_id="req", prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return LLMResult(content=json.dumps(payload), usage=usage, raw_response=None)


def test_valid_response_is_cached(tmp_path: Path) -> None:
    payload = {
        "proposals": [
            {"theta": 0.5, "depth": 2, "backend": "b"},
            {"theta": -0.5, "depth": 1, "backend": "a"},
        ]
    }
    generator = make_generator([result_from_payload(payload)], tmp_path=tmp_path)

    first = generator.propose_batch(2)
    assert first == payload["proposals"]

    second = generator.propose_batch(2)
    assert second != payload["proposals"]
    assert len(second) == 2
    stub = getattr(generator, "_test_provider")
    assert len(stub.temperatures) == 1


def test_logs_prompt_and_response(tmp_path: Path) -> None:
    payload = {
        "proposals": [
            {"theta": 0.1, "depth": 1, "backend": "a"},
            {"theta": -0.2, "depth": 2, "backend": "b"},
        ]
    }
    trace_path = tmp_path / "trace.jsonl"
    generator = make_generator(
        [result_from_payload(payload)],
        tmp_path=tmp_path,
        trace_path=trace_path,
    )

    proposals = generator.propose_batch(2)
    assert proposals == payload["proposals"]

    lines = [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["request"]["messages"]
    assert record["response"]["content"]
    assert record["response"]["usage"]["provider"] == "stub"


def test_invalid_json_falls_back_to_random(tmp_path: Path) -> None:
    responses = [
        LLMResult(content="not json", usage=None, raw_response=None),
        LLMResult(content="still not json", usage=None, raw_response=None),
    ]
    generator = make_generator(responses, tmp_path=tmp_path, max_retries=1)

    proposals = generator.propose_batch(2)
    assert len(proposals) == 2
    for proposal in proposals:
        assert -1.0 <= proposal["theta"] <= 1.0
        assert proposal["depth"] in {1, 2, 3}
        assert proposal["backend"] in {"a", "b", "c"}


def test_out_of_bounds_triggers_retry(tmp_path: Path) -> None:
    bad_payload = {"proposals": [{"theta": 5, "depth": 1, "backend": "a"}, {"theta": 5, "depth": 1, "backend": "a"}]}
    good_payload = {"proposals": [{"theta": 0.0, "depth": 1, "backend": "b"}, {"theta": 0.25, "depth": 2, "backend": "c"}]}
    responses = [result_from_payload(bad_payload), result_from_payload(good_payload)]
    generator = make_generator(responses, tmp_path=tmp_path, max_retries=2)

    proposals = generator.propose_batch(2)
    assert proposals == good_payload["proposals"]
    stub = getattr(generator, "_test_provider")
    assert len(stub.temperatures) == 2
    assert stub.temperatures[1] <= stub.temperatures[0]
    assert stub.temperatures[1] >= 0.2


def test_strategy_updates_cache_key(tmp_path: Path) -> None:
    payload = {"proposals": [{"theta": 0.1, "depth": 1, "backend": "a"}, {"theta": -0.1, "depth": 2, "backend": "b"}]}
    generator = make_generator([result_from_payload(payload)], tmp_path=tmp_path)

    initial_key = generator._cache_key(2)
    generator.update_strategy({"emphasis": "kl"})
    updated_key = generator._cache_key(2)
    assert initial_key != updated_key


def test_llm_only_parameter_passes_through(tmp_path: Path) -> None:
    payload = {
        "proposals": [
            {
                "theta": 0.1,
                "depth": 1,
                "backend": "a",
                "circuit_code": "OPENQASM 2.0;",
            }
        ]
    }
    generator = make_generator(
        [result_from_payload(payload)],
        tmp_path=tmp_path,
        search_space=make_search_space_with_llm_only(),
        batch_size=1,
    )

    proposals = generator.propose_batch(1)
    assert proposals[0]["circuit_code"] == "OPENQASM 2.0;"


def test_apply_search_space_overrides(tmp_path: Path) -> None:
    generator = make_generator([], tmp_path=tmp_path)

    updates = generator.apply_search_space_overrides(
        {
            "theta": {"low": -0.25, "high": 0.25},
            "backend": {"choices": ["a", "b"]},
        }
    )

    assert generator._search_space["theta"]["low"] == -0.25  # type: ignore[attr-defined]
    assert generator._search_space["theta"]["high"] == 0.25  # type: ignore[attr-defined]
    assert generator._search_space["backend"]["choices"] == ["a", "b"]  # type: ignore[attr-defined]
    assert any("theta" in entry for entry in updates)
