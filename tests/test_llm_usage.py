from __future__ import annotations

import csv
import json
from pathlib import Path

from astraia.llm_providers import (
    LLMResult,
    LLMUsage,
    LLMUsageLogger,
    Prompt,
    PromptMessage,
    LLMExchangeLogger,
)


def test_prompt_to_chat_messages_includes_system() -> None:
    prompt = Prompt(messages=[PromptMessage(role="user", content="hello")])
    converted = prompt.to_chat_messages(system="system role")
    assert converted[0] == {"role": "system", "content": "system role"}
    assert converted[1] == {"role": "user", "content": "hello"}


def test_llm_usage_logger_writes_headers(tmp_path: Path) -> None:
    log_path = tmp_path / "runs" / "exp" / "llm_usage.csv"
    logger = LLMUsageLogger(log_path)
    usage = LLMUsage(
        provider="openai",
        model="gpt",
        request_id="req-1",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    logger.log(usage)
    logger.log(usage)

    with log_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == [
        "timestamp",
        "provider",
        "model",
        "request_id",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ]
    assert len(rows) == 2
    assert rows[0]["provider"] == "openai"
    assert rows[0]["model"] == "gpt"


def test_llm_exchange_logger_supports_events(tmp_path: Path) -> None:
    log_path = tmp_path / "trace.jsonl"
    logger = LLMExchangeLogger(log_path)
    prompt = Prompt.from_text("hi")
    result = LLMResult(content="ok", usage=None, raw_response=None)

    logger.log(
        prompt=prompt,
        system="sys",
        tool=None,
        params={"temperature": 0.1},
        result=result,
        stage="guidance",
        trace_id="trace-1",
        latency_ms=12.5,
        parse={"status": "ok"},
        decision={"accepted": 1},
    )
    logger.log_event(
        kind="proposal_enqueued",
        stage="llm_guidance",
        trace_id="trace-1",
        data={"fingerprint": "abc"},
    )

    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[0]["stage"] == "guidance"
    assert lines[0]["trace_id"] == "trace-1"
    assert lines[0]["parse"]["status"] == "ok"
    assert lines[0]["decision"]["accepted"] == 1
    assert lines[1]["kind"] == "proposal_enqueued"
    assert lines[1]["event"]["fingerprint"] == "abc"
    assert lines[1]["trace_id"] == "trace-1"
