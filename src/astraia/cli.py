"""Command line interface for the Anemoi MVP skeleton."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Sequence

import yaml

try:  # pragma: no cover - prefer real pydantic
    from pydantic import ValidationError
except ImportError:  # pragma: no cover - offline fallback
    from ._compat.pydantic import ValidationError  # type: ignore[assignment]

from .config import OptimizationConfig
from .llm_guidance import create_llm_provider
from .llm_providers import ProviderUnavailableError
from .run_management import prepare_run_environment
from .tracking import RunMetadata
from .tracking import list_runs as tracking_list_runs
from .tracking import load_run as tracking_load_run
from .tracking import update_run_status

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
    subparsers = parser.add_subparsers(dest="command")
    _configure_runs_subcommands(subparsers)
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


def _add_runs_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Directory that stores run artifacts (default: runs).",
    )


def _configure_runs_subcommands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser] | None,
) -> None:
    if subparsers is None:
        return

    runs_parser = subparsers.add_parser(
        "runs",
        help="Inspect, monitor, and maintain experiment runs.",
        description="Inspect, monitor, and maintain experiment runs.",
    )
    runs_subparsers = runs_parser.add_subparsers(dest="runs_command", required=True)

    list_parser = runs_subparsers.add_parser(
        "list",
        help="List tracked runs in a compact table.",
        description="List tracked runs in a compact table.",
    )
    _add_runs_root_argument(list_parser)
    list_parser.add_argument(
        "--status",
        help="Filter runs by their current status.",
    )
    list_parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Apply additional filters using dotted keys (e.g. metadata.name=demo).",
    )
    list_parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of runs to display.",
    )
    list_parser.add_argument(
        "--sort",
        choices=("created", "run_id", "name", "status", "best_value"),
        default="created",
        help="Column to sort by (default: created).",
    )
    list_parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse the selected sort order.",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a table for downstream tooling.",
    )

    show_parser = runs_subparsers.add_parser(
        "show",
        help="Show metadata and resolved configuration for a run.",
        description="Show metadata and resolved configuration for a run.",
    )
    _add_runs_root_argument(show_parser)
    show_parser.add_argument("--run-id", required=True, help="Run identifier to inspect.")
    show_parser.add_argument(
        "--as-json",
        action="store_true",
        help="Emit combined metadata/configuration as JSON.",
    )

    delete_parser = runs_subparsers.add_parser(
        "delete",
        help="Delete a run directory and all associated artifacts.",
        description="Delete a run directory and all associated artifacts.",
    )
    _add_runs_root_argument(delete_parser)
    delete_parser.add_argument("--run-id", required=True, help="Run identifier to delete.")
    delete_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation prompts.",
    )
    delete_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be deleted without touching the filesystem.",
    )

    status_parser = runs_subparsers.add_parser(
        "status",
        help="Update run status metadata (best value, notes, metrics, ...).",
        description="Update run status metadata (best value, notes, metrics, ...).",
    )
    _add_runs_root_argument(status_parser)
    status_parser.add_argument("--run-id", required=True, help="Run identifier to update.")
    status_parser.add_argument(
        "--state",
        required=True,
        help="Status name to apply (e.g. running, completed, failed).",
    )
    status_parser.add_argument(
        "--note",
        help="Attach a short human readable note to the status payload.",
    )
    status_parser.add_argument(
        "--best-value",
        type=float,
        help="Record the best observed objective value for quick glance views.",
    )
    status_parser.add_argument(
        "--metric",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Attach additional metric values to the status payload.",
    )
    status_parser.add_argument(
        "--payload",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Arbitrary extra key/value pairs stored alongside the status.",
    )


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


def _handle_runs_command(args: argparse.Namespace) -> None:
    command = getattr(args, "runs_command", None)
    if command == "list":
        _runs_list_command(args)
    elif command == "show":
        _runs_show_command(args)
    elif command == "delete":
        _runs_delete_command(args)
    elif command == "status":
        _runs_status_command(args)
    else:  # pragma: no cover - argparse prevents this path
        raise SystemExit(f"Unknown runs sub-command: {command}")


def _runs_list_command(args: argparse.Namespace) -> None:
    filters: Dict[str, Any] = {}
    if getattr(args, "status", None):
        filters["status"] = args.status
    filter_pairs: Sequence[str] = getattr(args, "filter", []) or []
    filters.update(_parse_key_value_pairs(filter_pairs, flag="--filter"))
    runs = tracking_list_runs(filters or None, runs_root=args.runs_root)

    runs = _sort_runs(runs, key=args.sort, reverse=bool(args.reverse))
    limit = getattr(args, "limit", None)
    if isinstance(limit, int) and limit >= 0:
        runs = runs[:limit]

    if getattr(args, "json", False):
        payload = [_serialize_run_metadata(run) for run in runs]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if not runs:
        print(f"No runs found under {args.runs_root}.")
        return

    headers = ["run_id", "started_at", "problem_name", "best_value", "status"]
    rows: list[Sequence[str]] = []
    for metadata in runs:
        rows.append(
            [
                metadata.run_id,
                _format_datetime(metadata.created_at),
                str(metadata.metadata.get("name") or metadata.metadata.get("problem") or "-"),
                _format_best_value(_extract_best_value(metadata.status_payload)),
                metadata.status or "-",
            ]
        )

    print(_render_table(headers, rows))


def _runs_show_command(args: argparse.Namespace) -> None:
    metadata = tracking_load_run(args.run_id, runs_root=args.runs_root)
    config = _load_config_artifact(metadata)

    if getattr(args, "as_json", False):
        payload = {"meta": metadata.raw, "config_resolved": config}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print(_render_run_summary(metadata, config))


def _runs_delete_command(args: argparse.Namespace) -> None:
    metadata = tracking_load_run(args.run_id, runs_root=args.runs_root)
    run_dir = metadata.run_dir
    if not run_dir.exists():
        print(f"Run '{metadata.run_id}' already deleted ({run_dir}).")
        return

    resolved_root = Path(args.runs_root).resolve()
    resolved_dir = run_dir.resolve()
    try:
        resolved_dir.relative_to(resolved_root)
    except ValueError:
        raise SystemExit(
            f"Refusing to delete run outside runs root: {resolved_dir} (root={resolved_root})."
        )

    if getattr(args, "dry_run", False):
        print(f"[dry-run] Would delete {resolved_dir}")
        return

    if not getattr(args, "yes", False):
        response = input(f"Delete run '{metadata.run_id}' at {resolved_dir}? [y/N]: ").strip().lower()
        if response not in {"y", "yes"}:
            print("Aborted.")
            return

    shutil.rmtree(resolved_dir)
    print(f"Deleted run '{metadata.run_id}' at {resolved_dir}.")


def _runs_status_command(args: argparse.Namespace) -> None:
    payload: Dict[str, Any] = _parse_key_value_pairs(
        getattr(args, "payload", []) or [],
        flag="--payload",
    )
    metrics = _parse_key_value_pairs(
        getattr(args, "metric", []) or [],
        flag="--metric",
    )
    if metrics:
        payload["metrics"] = metrics
    if getattr(args, "note", None):
        payload["note"] = args.note
    if getattr(args, "best_value", None) is not None:
        payload["best_value"] = args.best_value

    updated = update_run_status(
        args.run_id,
        args.state,
        runs_root=args.runs_root,
        **payload,
    )
    print(f"Run '{updated.run_id}' status updated to {updated.status}.")


def _parse_key_value_pairs(items: Sequence[str], *, flag: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for raw in items:
        if "=" not in raw:
            raise SystemExit(f"{flag} expects KEY=VALUE pairs, got: {raw}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"{flag} expects non-empty keys (got: {raw}).")
        parsed[key] = _coerce_scalar(value.strip())
    return parsed


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _sort_runs(
    runs: Sequence[RunMetadata], *, key: str, reverse: bool
) -> list[RunMetadata]:
    def _key(metadata: RunMetadata) -> Any:
        if key == "run_id":
            return metadata.run_id
        if key == "name":
            return metadata.metadata.get("name") or ""
        if key == "status":
            return metadata.status or ""
        if key == "best_value":
            return _extract_best_value(metadata.status_payload) or 0.0
        return metadata.created_at or datetime.fromtimestamp(0, tz=timezone.utc)

    return sorted(runs, key=_key, reverse=reverse)


def _extract_best_value(status_payload: Mapping[str, Any]) -> Any:
    if isinstance(status_payload, Mapping):
        value = status_payload.get("best_value")
        if isinstance(value, (int, float)):
            return value
        return value
    return None


def _format_best_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def _format_datetime(value: datetime | None) -> str:
    if value is None:
        return "-"
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone().strftime("%Y-%m-%d %H:%M")


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    header_line = " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    separator = "-+-".join("-" * width for width in widths)
    body = [" | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) for row in rows]
    return "\n".join([header_line, separator, *body])


def _serialize_run_metadata(metadata: "RunMetadata") -> Dict[str, Any]:
    return {
        "run_id": metadata.run_id,
        "run_dir": str(metadata.run_dir),
        "created_at": metadata.created_at.isoformat() if metadata.created_at else None,
        "status": metadata.status,
        "status_payload": metadata.status_payload,
        "metadata": dict(metadata.metadata),
        "report": dict(metadata.report),
        "artifacts": dict(metadata.artifacts),
        "source": dict(metadata.source),
    }


def _load_config_artifact(metadata: "RunMetadata") -> Dict[str, Any] | None:
    config_path = metadata.artifact_path("config_resolved")
    if config_path and config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _render_run_summary(metadata: "RunMetadata", config: Mapping[str, Any] | None) -> str:
    lines = [
        f"Run ID         : {metadata.run_id}",
        f"Directory      : {metadata.run_dir}",
        f"Created at     : {_format_datetime(metadata.created_at)}",
        f"Status         : {metadata.status or '-'}",
    ]

    payload_lines = []
    for key, value in sorted(metadata.status_payload.items()):
        if key == "state":
            continue
        payload_lines.append(f"  - {key}: {value}")
    if payload_lines:
        lines.append("Status payload :")
        lines.extend(payload_lines)

    if metadata.metadata:
        lines.append("Metadata       :")
        for key, value in metadata.metadata.items():
            lines.append(f"  - {key}: {value}")

    if metadata.report:
        lines.append("Report         :")
        for key, value in metadata.report.items():
            lines.append(f"  - {key}: {value}")

    artifact_paths = metadata.artifact_paths
    if artifact_paths:
        lines.append("Artifacts      :")
        for key, path in sorted(artifact_paths.items()):
            lines.append(f"  - {key}: {path}")

    if metadata.source:
        lines.append("Source         :")
        for key, value in metadata.source.items():
            lines.append(f"  - {key}: {value}")

    if config:
        lines.append("")
        lines.append("[Resolved configuration]")
        lines.append(summarize_config(dict(config)))

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    if getattr(args, "command", None) == "runs":
        _handle_runs_command(args)
        return

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
