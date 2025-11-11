from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from typing import Any, List

from astraia.llm_guidance import LLMProposalGenerator, PromptCache
from astraia.llm_providers import LLMResult, LLMUsage, LLMUsageLogger


class StubProvider:
    def __init__(self, responses: List[LLMResult]) -> None:
        self._responses = deque(responses)
        self.temperatures: List[float | None] = []

    def generate(self, prompt, *, temperature=None, json_mode=False, system=None):  # noqa: ANN001
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


def make_generator(
    responses: List[LLMResult],
    *,
    tmp_path: Path,
    base_temperature: float = 0.6,
    min_temperature: float = 0.2,
    max_retries: int = 2,
) -> LLMProposalGenerator:
    provider = StubProvider(responses)
    usage_logger = LLMUsageLogger(tmp_path / "usage.csv")
    generator = LLMProposalGenerator(
        search_space=make_search_space(),
        problem_summary="demo",
        objective="minimize",
        batch_size=2,
        base_temperature=base_temperature,
        min_temperature=min_temperature,
        max_retries=max_retries,
        provider=provider,
        usage_logger=usage_logger,
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
    assert second == payload["proposals"]
    stub = getattr(generator, "_test_provider")
    assert len(stub.temperatures) == 1


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

