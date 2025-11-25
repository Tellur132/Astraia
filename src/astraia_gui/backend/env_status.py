from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Tuple

DEFAULT_ENV_REQUIREMENTS: Dict[str, Tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY",),
}


def env_status() -> dict[str, Any]:
    """Return presence/absence of required environment keys."""
    requirements = _load_env_requirements()
    providers: dict[str, dict[str, Any]] = {}
    missing_keys: list[str] = []

    for provider, keys in requirements.items():
        present = []
        missing = []
        for key in keys:
            value = os.getenv(key)
            if value:
                present.append({"name": key, "masked": _mask_secret(value)})
            else:
                missing.append(key)
                missing_keys.append(key)
        providers[provider] = {"ok": not missing, "missing": missing, "present": present}

    return {"ok": not missing_keys, "missing_keys": missing_keys, "providers": providers}


def _load_env_requirements() -> Dict[str, Tuple[str, ...]]:
    try:  # Lazy import to avoid CLI startup side-effects
        from astraia.cli import _ENV_REQUIREMENTS as cli_requirements  # type: ignore[attr-defined]
    except Exception:
        return dict(DEFAULT_ENV_REQUIREMENTS)

    merged: Dict[str, Tuple[str, ...]] = dict(DEFAULT_ENV_REQUIREMENTS)
    merged.update(cli_requirements)
    return merged


def _mask_secret(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) <= 4:
        return "*" * len(cleaned)
    return f"{cleaned[:2]}{'*' * max(len(cleaned) - 4, 0)}{cleaned[-2:]}"

