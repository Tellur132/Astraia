from __future__ import annotations

import csv
from pathlib import Path

from astraia.llm_providers import LLMUsage, LLMUsageLogger, Prompt, PromptMessage


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
