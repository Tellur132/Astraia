"""Command line interface for the Anemoi MVP skeleton."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping

import yaml

try:  # pragma: no cover - prefer real pydantic
    from pydantic import ValidationError
except ImportError:  # pragma: no cover - offline fallback
    from ._compat.pydantic import ValidationError  # type: ignore[assignment]

from .config import OptimizationConfig
from .llm_guidance import create_llm_provider
from .llm_providers import ProviderUnavailableError
from .run_management import prepare_run_environment

if TYPE_CHECKING:
    from .optimization import OptimizationResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the qGAN KL optimization loop or inspect the configuration summary."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qgan_kl.yaml"),
        help="Path to the optimization configuration YAML file.",
    )
    parser.add_argument(
        "--as-json",
        action="store_true",
        help="Print the validated configuration as JSON for downstream tooling.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Only print a summary of the configuration without running the loop.",
    )
    parser.add_argument(
        "--planner",
        choices=("none", "rule", "llm"),
        help="Override planner backend (default: use value from config).",
    )
    parser.add_argument(
        "--planner-config",
        type=Path,
        help="Path to planner-specific configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate configuration and ping the configured LLM provider without running the"
            " optimization loop."
        ),
    )
    return parser.parse_args()


def load_config(path: Path) -> OptimizationConfig:
    if not path.exists():
        raise SystemExit(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise SystemExit("Configuration root must be a mapping (YAML dictionary).")

    try:
        validated = OptimizationConfig.model_validate(data)
    except ValidationError as exc:  # pragma: no cover - exercised via CLI tests
        details = []
        for error in exc.errors(include_url=False):
            location = ".".join(str(loc) for loc in error["loc"])
            details.append(f"- {location or '<root>'}: {error['msg']}")
        message = "Configuration validation failed:\n" + "\n".join(details)
        raise SystemExit(message) from exc

    return validated


def apply_planner_overrides(
    config: OptimizationConfig,
    *,
    planner: str | None,
    planner_config: Path | None,
) -> OptimizationConfig:
    if planner is None and planner_config is None:
        return config

    data = config.model_dump(mode="python")

    if planner == "none":
        data["planner"] = None
    elif planner is not None:
        existing = dict(data.get("planner") or {})
        existing["backend"] = planner
        existing["enabled"] = True
        data["planner"] = existing

    if planner_config is not None:
        if not planner_config.exists():
            raise SystemExit(f"Planner configuration file not found: {planner_config}")
        planner_section = data.get("planner")
        if planner_section is None:
            raise SystemExit("--planner-config requires a planner to be enabled")
        updated = dict(planner_section)
        updated["config_path"] = str(planner_config)
        data["planner"] = updated

    try:
        return OptimizationConfig.model_validate(data)
    except ValidationError as exc:  # pragma: no cover - exercised via CLI tests
        details = []
        for error in exc.errors(include_url=False):
            location = ".".join(str(loc) for loc in error["loc"])
            details.append(f"- {location or '<root>'}: {error['msg']}")
        message = "Configuration validation failed after applying planner overrides:\n" + "\n".join(details)
        raise SystemExit(message) from exc


_ENV_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY",),
}


def parse_env_file(path: Path) -> Dict[str, str]:
    """Parse key-value pairs from a dotenv-style file."""

    values: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            key = key.strip()
            value = raw_value.strip()
            if (value.startswith("\"") and value.endswith("\"")) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            values[key] = value
    return values


def ensure_env_keys(
    llm_cfg: Mapping[str, Any] | None,
    *,
    env_path: Path,
) -> Dict[str, str]:
    """Validate that required secrets for the configured provider exist in the .env file."""

    if llm_cfg is None:
        return {}

    provider = str(llm_cfg.get("provider", "")).strip().lower()
    if not provider:
        return {}

    required = _ENV_REQUIREMENTS.get(provider)
    if not required:
        return {}

    if not env_path.exists():
        raise SystemExit(
            f"LLM provider '{provider}' requires {', '.join(required)} defined in {env_path}."
        )

    env_values = parse_env_file(env_path)
    loaded: Dict[str, str] = {}
    missing: list[str] = []
    for key in required:
        value = env_values.get(key, "").strip()
        if value:
            loaded[key] = value
        else:
            missing.append(key)

    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing required secrets in {env_path} for provider '{provider}': {joined}."
        )

    for key, value in loaded.items():
        if not os.environ.get(key):
            os.environ[key] = value

    return loaded


def ping_llm_provider(llm_cfg: Mapping[str, Any] | None) -> None:
    """Instantiate the configured provider and perform a connectivity check."""

    if llm_cfg is None:
        print("Dry run: no LLM provider configured; nothing to ping.")
        return

    provider_name = str(llm_cfg.get("provider", "")).strip().lower()
    model_name = str(llm_cfg.get("model", "")).strip()

    if not provider_name:
        print("Dry run: LLM provider not specified; nothing to ping.")
        return

    try:
        provider, _ = create_llm_provider(llm_cfg, strict=True)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    except ProviderUnavailableError as exc:
        raise SystemExit(f"LLM provider '{provider_name}' unavailable: {exc}") from exc

    if provider is None:
        raise SystemExit(
            f"LLM provider '{provider_name}' could not be instantiated for ping operations."
        )

    ping_method = getattr(provider, "ping", None)
    if ping_method is None:
        raise SystemExit(f"LLM provider '{provider_name}' does not support connectivity checks.")

    try:
        ping_method()
    except Exception as exc:  # pragma: no cover - dependent on network and SDK behaviour
        raise SystemExit(f"Failed to ping LLM provider '{provider_name}': {exc}") from exc

    descriptor = model_name or "<unknown model>"
    print(f"Dry run: successfully pinged provider '{provider_name}' with model '{descriptor}'.")


def summarize_config(config: Dict[str, Any]) -> str:
    metadata = config.get("metadata", {})
    search = config.get("search", {})
    stopping = config.get("stopping", {})
    report = config.get("report", {})

    direction = search.get("direction", "N/A")
    if isinstance(direction, list):
        direction_display = ", ".join(str(item) for item in direction)
    else:
        direction_display = str(direction)
    metric = search.get("metric", "kl")
    if isinstance(metric, list):
        metric_display = ", ".join(str(item) for item in metric)
    else:
        metric_display = str(metric)

    lines = [
        f"Experiment name : {metadata.get('name', 'N/A')}",
        f"Description    : {metadata.get('description', 'N/A')}",
        "",
        "[Search]",
        f"  Library      : {search.get('library', 'N/A')}",
        f"  Sampler      : {search.get('sampler', 'N/A')}",
        f"  Trials       : {search.get('n_trials', 'N/A')}",
        f"  Direction    : {direction_display}",
        f"  Metric       : {metric_display}",
        "",
        "[Stopping]",
        f"  max_trials   : {stopping.get('max_trials', 'N/A')}",
        f"  max_minutes  : {stopping.get('max_time_minutes', 'N/A')}",
        f"  patience     : {stopping.get('no_improve_patience', 'N/A')}",
        "",
        "[Report]",
        f"  metrics      : {', '.join(report.get('metrics', [])) if report.get('metrics') else 'N/A'}",
        f"  output_dir   : {report.get('output_dir', 'reports')}",
    ]

    return "\n".join(lines)


def format_result(result: "OptimizationResult") -> str:
    lines = [
        "Optimization finished.",
        f"Trials executed : {result.trials_completed}",
        f"Best value      : {result.best_value}",
        "Best parameters :",
    ]
    lines.extend([f"  - {name}: {value}" for name, value in result.best_params.items()])
    lines.append("Best metrics    :")
    lines.extend([f"  - {name}: {value}" for name, value in result.best_metrics.items()])
    if result.total_cost is not None:
        lines.append(f"Total cost      : {result.total_cost}")
    if result.hypervolume is not None:
        lines.append(f"Hypervolume     : {result.hypervolume}")
    if result.pareto_front:
        lines.append(f"Pareto points   : {len(result.pareto_front)}")
    if result.early_stopped_reason:
        lines.append(f"Early stop      : {result.early_stopped_reason}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.dry_run and (args.as_json or args.summarize):
        raise SystemExit("--dry-run cannot be combined with --as-json or --summarize")

    config_model = load_config(args.config)
    config_model = apply_planner_overrides(
        config_model,
        planner=args.planner,
        planner_config=args.planner_config,
    )
    config = config_model.model_dump(mode="python")

    env_file = Path(os.environ.get("ASTRAIA_ENV_FILE", ".env"))
    ensure_env_keys(config.get("llm"), env_path=env_file)

    if args.as_json:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return

    if args.summarize:
        print(summarize_config(config))
        return

    if args.dry_run:
        ping_llm_provider(config.get("llm"))
        return

    from .optimization import run_optimization

    config_for_run, _ = prepare_run_environment(
        config_model=config_model,
        config_source=args.config,
    )

    result = run_optimization(config_for_run)
    print(format_result(result))


if __name__ == "__main__":
    main()
