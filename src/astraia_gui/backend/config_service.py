from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml
from astraia.config import OptimizationConfig

CONFIG_EXTENSIONS = {".yaml", ".yml"}
CONFIG_ROOT_ENV = "ASTRAIA_CONFIG_ROOT"


class ConfigLoadError(Exception):
    """Raised when a configuration file cannot be read or parsed."""


@lru_cache(maxsize=1)
def get_config_root() -> Path:
    """Locate the configs/ directory used by the GUI backend.

    Order of precedence:
    1. ``ASTRAIA_CONFIG_ROOT`` environment variable
    2. ``./configs`` relative to current working directory
    3. ``configs`` next to the source tree (repo root)
    """

    candidates: list[Path] = []

    env_root = os.getenv(CONFIG_ROOT_ENV)
    if env_root:
        candidates.append(Path(env_root).expanduser())

    cwd_root = Path.cwd() / "configs"
    candidates.append(cwd_root)

    source_root = Path(__file__).resolve().parents[3] / "configs"
    candidates.append(source_root)

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No configs directory found. Searched: {searched}")


def list_config_entries(root: Path | None = None) -> list[dict[str, Any]]:
    """Return available YAML configs with basic metadata."""
    config_root = root or get_config_root()
    entries: list[dict[str, Any]] = []

    for path in sorted(config_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in CONFIG_EXTENSIONS:
            continue

        rel_path = path.relative_to(config_root)
        entries.append(
            {
                "name": rel_path.with_suffix("").as_posix(),
                "path": rel_path.as_posix(),
                "tags": _derive_tags(rel_path),
            }
        )

    return entries


def read_config_text(config_path: str, root: Path | None = None) -> tuple[Path, str]:
    """Return the resolved path and YAML text for a config."""
    resolved = _resolve_config_path(config_path, root=root)

    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigLoadError(f"Failed to read config: {resolved}") from exc

    return resolved, text


def load_config_data(config_path: str, root: Path | None = None) -> tuple[Path, Dict[str, Any]]:
    """Load a config as a Python mapping (without validation)."""
    resolved, text = read_config_text(config_path, root=root)

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"Failed to parse YAML in {resolved}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigLoadError(f"Configuration root must be a mapping: {resolved}")

    return resolved, data


def validate_config(config_path: str, root: Path | None = None) -> tuple[Path, OptimizationConfig]:
    """Load and validate a config into an OptimizationConfig model."""
    resolved, data = load_config_data(config_path, root=root)
    return resolved, OptimizationConfig.model_validate(data)


def build_config_summary(model: OptimizationConfig) -> dict[str, Any]:
    """Build a structured summary used by the GUI."""
    search = model.search
    stopping = model.stopping
    report = model.report

    summary = {
        "metadata": {
            "name": model.metadata.name,
            "description": model.metadata.description,
        },
        "search": {
            "library": search.library,
            "sampler": search.sampler,
            "n_trials": search.n_trials,
            "multi_objective": bool(search.multi_objective),
            "metrics": list(search.metric_names),
            "directions": list(search.direction_names),
        },
        "stopping": {
            "max_trials": stopping.max_trials,
            "max_time_minutes": stopping.max_time_minutes,
            "no_improve_patience": stopping.no_improve_patience,
            "cost_metric": stopping.cost_metric,
            "max_total_cost": stopping.max_total_cost,
        },
        "report": {
            "metrics": list(report.metrics),
            "output_dir": report.output_dir,
            "filename": report.filename,
        },
        "search_space": _summarize_search_space(model.search_space),
        "features": {
            "planner_enabled": bool(model.planner and model.planner.enabled),
            "llm_enabled": bool(model.llm is not None),
            "llm_guidance_enabled": bool(model.llm_guidance and model.llm_guidance.enabled),
            "meta_search_enabled": bool(model.meta_search and model.meta_search.enabled),
            "llm_critic_enabled": bool(model.llm_critic and model.llm_critic.enabled),
        },
    }

    return summary


def _derive_tags(path: Path) -> list[str]:
    """Use parent directories as lightweight tags."""
    return [part for part in path.parent.parts if part]


def _resolve_config_path(config_path: str, root: Path | None = None) -> Path:
    """Resolve a config path safely under the config root."""
    config_root = (root or get_config_root()).resolve()
    raw_path = Path(config_path.strip()).as_posix()

    candidates = _candidate_paths(raw_path, config_root)
    for candidate in candidates:
        try:
            candidate.relative_to(config_root)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Config '{config_path}' not found under {config_root}")


def _candidate_paths(raw_path: str, root: Path) -> Iterable[Path]:
    path_obj = Path(raw_path)
    if path_obj.suffix:
        yield (root / path_obj).resolve()
        return

    yield (root / path_obj).with_suffix(".yaml").resolve()
    yield (root / path_obj).with_suffix(".yml").resolve()


def _summarize_search_space(space: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for name, spec in space.items():
        entry: dict[str, Any] = {"name": name}
        for key, value in spec.items():
            entry[key] = value
        entries.append(entry)
    return entries
