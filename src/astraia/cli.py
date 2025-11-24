"""Command line interface for the Anemoi MVP skeleton."""
from __future__ import annotations

import argparse
import csv
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
from .run_summary import summarize_run_results
from .tracking import RunMetadata
from .tracking import list_runs as tracking_list_runs
from .tracking import load_run as tracking_load_run
from .tracking import update_run_status
from .visualization import ObjectiveSpec, VisualizationError, plot_history, plot_pareto_front

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
    _configure_visualize_subcommand(subparsers)
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
    status_parser.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Attach comparison tags such as multi_objective or objectives (repeatable).",
    )
    status_parser.add_argument(
        "--pareto-summary",
        type=Path,
        metavar="PATH",
        help="Path to a JSON document describing the Pareto front overview.",
    )

    diff_parser = runs_subparsers.add_parser(
        "diff",
        help="Compare resolved configuration differences across runs.",
        description="Compare resolved configuration differences across runs.",
    )
    _add_runs_root_argument(diff_parser)
    diff_parser.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        required=True,
        metavar="RUN_ID",
        help="Run identifier to include in the comparison (repeat for multiple runs).",
    )

    compare_parser = runs_subparsers.add_parser(
        "compare",
        help="Compare key metrics across completed runs.",
        description="Compare key metrics across completed runs.",
    )
    _add_runs_root_argument(compare_parser)
    compare_parser.add_argument(
        "--runs",
        nargs="+",
        metavar="RUN_ID",
        required=True,
        help="Run identifiers to include in the comparison table.",
    )
    compare_parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        metavar="NAME",
        help="Restrict the comparison to specific metric names (default: all).",
    )
    compare_parser.add_argument(
        "--stat",
        action="append",
        dest="stats",
        metavar="NAME",
        choices=("best", "median", "mean"),
        help="Summary statistic to include (repeat for multiple; default: best, median, mean).",
    )
    compare_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a formatted table.",
    )


def _configure_visualize_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser] | None,
) -> None:
    if subparsers is None:
        return

    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Render quick visualizations for a completed run.",
        description="Render quick visualizations for a completed run.",
    )
    _add_runs_root_argument(visualize_parser)
    visualize_parser.add_argument("--run-id", required=True, help="Run identifier to visualize.")
    visualize_parser.add_argument(
        "--type",
        choices=("pareto", "history"),
        default="history",
        help="Visualization type to render (default: history).",
    )
    visualize_parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        metavar="NAME",
        help="Metric/objective name to include (repeatable).",
    )
    visualize_parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path for the rendered image (default: run directory).",
    )
    visualize_parser.add_argument(
        "--title",
        help="Optional title override for the generated plot.",
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

    metrics, directions, multi_objective = _extract_objective_settings(search)
    metric_display = ", ".join(metrics) if metrics else "N/A"
    direction_pairs = [
        f"{direction}: {metric}"
        for metric, direction in zip(metrics, directions, strict=False)
    ]
    direction_display = ", ".join(direction_pairs) if direction_pairs else "N/A"
    multi_display = "yes" if multi_objective else "no"

    lines = [
        f"Experiment name : {metadata.get('name', 'N/A')}",
        f"Description    : {metadata.get('description', 'N/A')}",
        "",
        "[Search]",
        f"  Library      : {search.get('library', 'N/A')}",
        f"  Sampler      : {search.get('sampler', 'N/A')}",
        f"  Trials       : {search.get('n_trials', 'N/A')}",
        f"  Multi-objective : {multi_display}",
        f"  Metrics      : {metric_display}",
        f"  Directions   : {direction_display}",
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


def _extract_objective_settings(search: Mapping[str, Any]) -> tuple[list[str], list[str], bool]:
    metrics, directions = _parse_metric_definitions(search)
    multi_objective = bool(search.get("multi_objective")) or len(metrics) > 1
    return metrics, directions, multi_objective


def _ensure_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value]
    return []


def _default_visualization_path(run_dir: Path, viz_type: str) -> Path:
    name = f"visualization_{viz_type}.png"
    return run_dir / name


def _extract_metric_names_from_config(config: Mapping[str, Any] | None) -> list[str]:
    if not config:
        return []
    search = config.get("search")
    if not isinstance(search, Mapping):
        return []
    metrics, _ = _parse_metric_definitions(search)
    return metrics


def _extract_metric_names_from_log(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    with log_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return []
    return [col.replace("metric_", "", 1) for col in header if col.startswith("metric_")]


def _build_objective_specs(
    metric_names: Sequence[str], config: Mapping[str, Any] | None
) -> list[ObjectiveSpec]:
    directions = _resolve_visualization_directions(config, metric_names)
    return [ObjectiveSpec(name=name, direction=directions.get(name, "minimize")) for name in metric_names]


def _resolve_visualization_directions(
    config: Mapping[str, Any] | None, metric_names: Sequence[str]
) -> Dict[str, str]:
    directions = {name: "minimize" for name in metric_names}
    if not config:
        return directions
    search = config.get("search")
    if not isinstance(search, Mapping):
        return directions

    metric_list, direction_list = _parse_metric_definitions(search)

    for metric, direction in zip(metric_list, direction_list, strict=False):
        directions[str(metric)] = str(direction).lower()
    return directions


def _parse_metric_definitions(search: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    metrics_value = search.get("metrics")
    if metrics_value is None:
        metrics_value = search.get("metric")
    directions_value = search.get("directions")
    if directions_value is None:
        directions_value = search.get("direction")

    entries: list[Any]
    if isinstance(metrics_value, Sequence) and not isinstance(
        metrics_value, (str, bytes, bytearray)
    ):
        entries = list(metrics_value)
    elif metrics_value is None:
        entries = []
    else:
        entries = [metrics_value]

    metrics: list[str] = []
    directions: list[str] = []
    for idx, entry in enumerate(entries):
        name, explicit_direction = _parse_metric_entry(entry)
        metrics.append(name)
        directions.append(
            _normalise_direction_value(
                explicit_direction,
                default_source=directions_value,
                idx=idx,
                total=len(entries),
            )
        )

    return metrics, directions


def _parse_metric_entry(entry: Any) -> tuple[str, str | None]:
    if isinstance(entry, Mapping):
        name = str(entry.get("name", "")).strip()
        if not name:
            raise SystemExit("search.metrics entries must include a non-empty 'name'")
        direction = entry.get("direction")
        return name, str(direction) if direction is not None else None

    name = str(entry).strip()
    if not name:
        raise SystemExit("search.metric entries must be non-empty strings")
    return name, None


def _normalise_direction_value(
    explicit: str | None,
    *,
    default_source: Any,
    idx: int,
    total: int,
) -> str:
    direction_value: Any
    if explicit is not None:
        direction_value = explicit
    elif isinstance(default_source, Sequence) and not isinstance(
        default_source, (str, bytes, bytearray)
    ):
        if not default_source:
            raise SystemExit("search.directions must not be empty when provided")
        if len(default_source) == 1:
            direction_value = default_source[0]
        elif idx < len(default_source):
            direction_value = default_source[idx]
        else:
            raise SystemExit("search.directions must match the number of metrics")
    else:
        direction_value = default_source or "minimize"

    direction = str(direction_value).lower().strip()
    if direction not in {"minimize", "maximize"}:
        raise SystemExit("search.direction entries must be 'minimize' or 'maximize'")
    return direction


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
    elif command == "diff":
        _runs_diff_command(args)
    elif command == "compare":
        _runs_compare_command(args)
    else:  # pragma: no cover - argparse prevents this path
        raise SystemExit(f"Unknown runs sub-command: {command}")


def _handle_visualize_command(args: argparse.Namespace) -> None:
    metadata = tracking_load_run(args.run_id, runs_root=args.runs_root)
    log_path = metadata.artifact_path("log") or metadata.run_dir / "log.csv"
    config = _load_config_artifact(metadata)

    requested_metrics: Sequence[str] = getattr(args, "metrics", []) or []
    metrics = list(requested_metrics) or _extract_metric_names_from_config(config)
    if not metrics:
        metrics = _extract_metric_names_from_log(log_path)
    if not metrics:
        raise SystemExit("No metrics found in the run configuration or log file.")

    objectives = _build_objective_specs(metrics, config)
    title = getattr(args, "title", None)
    output_path = getattr(args, "output", None)
    resolved_output = output_path or _default_visualization_path(metadata.run_dir, args.type)

    try:
        if args.type == "pareto":
            if len(objectives) < 2:
                raise SystemExit("Pareto visualizations require at least two metrics.")
            result_path = plot_pareto_front(
                log_path,
                objectives[:2],
                title=title,
                output_path=resolved_output,
            )
        else:
            result_path = plot_history(
                log_path,
                objectives,
                title=title,
                output_path=resolved_output,
            )
    except VisualizationError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Visualization written to {result_path}")


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
    metadata = tracking_load_run(args.run_id, runs_root=args.runs_root)
    comparison_path = metadata.run_dir / "comparison_summary.json"
    existing_comparison = _load_comparison_summary(comparison_path)
    artifacts: Dict[str, str] = {"comparison_summary": str(comparison_path)}

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
    tags = _parse_key_value_pairs(
        getattr(args, "tag", []) or [],
        flag="--tag",
    )
    if tags:
        payload["tags"] = tags
    if getattr(args, "note", None):
        payload["note"] = args.note
    if getattr(args, "best_value", None) is not None:
        payload["best_value"] = args.best_value

    pareto_summary_path: Path | None = getattr(args, "pareto_summary", None)
    pareto_payload: Any | None = None
    if pareto_summary_path is not None:
        pareto_payload = _load_json_document(pareto_summary_path)

    updated = update_run_status(
        args.run_id,
        args.state,
        runs_root=args.runs_root,
        artifacts=artifacts,
        **payload,
    )
    _write_comparison_summary(
        comparison_path,
        updated,
        pareto_summary=pareto_payload,
        existing=existing_comparison,
    )
    print(f"Run '{updated.run_id}' status updated to {updated.status}.")


_MISSING = object()


def _load_json_document(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"Pareto summary file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:  # pragma: no cover - exercised via CLI tests
        raise SystemExit(f"Failed to parse JSON from {path}: {exc}") from exc


def _load_comparison_summary(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        return None
    if isinstance(data, Mapping):
        return data
    return None


def _write_comparison_summary(
    path: Path,
    metadata: "RunMetadata",
    *,
    pareto_summary: Any | None,
    existing: Mapping[str, Any] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tags_payload = metadata.status_payload.get("tags")
    tags: Any
    if isinstance(tags_payload, Mapping):
        tags = dict(tags_payload)
    else:
        tags = tags_payload
    metrics_payload = metadata.status_payload.get("metrics")
    metrics: Any
    if isinstance(metrics_payload, Mapping):
        metrics = dict(metrics_payload)
    else:
        metrics = metrics_payload
    summary: Dict[str, Any] = {
        "run_id": metadata.run_id,
        "state": metadata.status,
        "best_value": metadata.status_payload.get("best_value"),
        "metrics": metrics,
        "tags": tags,
        "updated_at": metadata.status_updated_at.isoformat()
        if metadata.status_updated_at
        else None,
    }
    previous_summary = existing.get("pareto_summary") if existing else None
    final_pareto_summary = pareto_summary if pareto_summary is not None else previous_summary
    if final_pareto_summary is not None:
        summary["pareto_summary"] = final_pareto_summary
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _runs_diff_command(args: argparse.Namespace) -> None:
    run_ids: Sequence[str] = getattr(args, "run_ids", []) or []
    if len(run_ids) < 2:
        raise SystemExit("--run-id must be provided at least twice to compare runs")

    configs: Dict[str, Mapping[str, Any]] = {}
    run_dirs: Dict[str, str] = {}
    for run_id in run_ids:
        metadata = tracking_load_run(run_id, runs_root=args.runs_root)
        config = _load_config_artifact(metadata)
        if config is None:
            raise SystemExit(
                f"Run '{run_id}' does not have a resolved configuration artifact to diff."
            )
        configs[run_id] = config
        run_dirs[run_id] = str(metadata.run_dir)

    rendered = _render_config_diff(
        configs,
        ordered_ids=list(run_ids),
        run_dirs=run_dirs,
    )
    if rendered:
        print(rendered)
    else:
        print("No configuration differences found between the selected runs.")


def _runs_compare_command(args: argparse.Namespace) -> None:
    run_ids: Sequence[str] = getattr(args, "runs", []) or []
    if not run_ids:
        raise SystemExit("--runs expects at least one run identifier")

    metrics = getattr(args, "metrics", None)
    stats = getattr(args, "stats", None)

    summaries: list[Dict[str, Any]] = []
    for run_id in run_ids:
        metadata = tracking_load_run(run_id, runs_root=args.runs_root)
        config = _load_config_artifact(metadata)
        summary = summarize_run_results(
            metadata,
            config=config,
            metrics=metrics,
            statistic_names=stats,
        )
        summaries.append(summary)

    if getattr(args, "json", False):
        print(json.dumps(summaries, indent=2, ensure_ascii=False))
        return

    stat_labels: Sequence[str] = stats or ("best", "median", "mean")
    headers = [
        "run_id",
        "metric",
        *stat_labels,
        "n_trials",
        "valid_trials",
        "early_stop",
    ]
    rows: list[list[str]] = []
    for summary in summaries:
        metrics_payload = summary.get("metrics") or {}
        metric_items = list(metrics_payload.items())
        if not metric_items:
            metric_items = [("-", {})]
        for idx, (metric_name, values) in enumerate(metric_items):
            row = [
                summary["run_id"] if idx == 0 else "",
                metric_name,
            ]
            for label in stat_labels:
                row.append(_format_best_value(values.get(label)))
            if idx == 0:
                row.extend(
                    [
                        str(summary.get("n_trials", 0)),
                        str(summary.get("n_valid_trials", 0)),
                        summary.get("early_stop_reason") or "-",
                    ]
                )
            else:
                row.extend(["", "", ""])
            rows.append(row)

    if not rows:
        print("No metrics available for the selected runs.")
        return

    print(_render_table(headers, rows))


def _render_config_diff(
    configs: Mapping[str, Mapping[str, Any]], *, ordered_ids: Sequence[str], run_dirs: Mapping[str, str]
) -> str | None:
    flattened = {run_id: _flatten_config_for_diff(config) for run_id, config in configs.items()}
    all_keys: list[str] = sorted({key for mapping in flattened.values() for key in mapping})
    if not all_keys:
        return None

    rows: list[list[str]] = []
    for key in all_keys:
        normalized: list[Any] = []
        display: list[str] = []
        for run_id in ordered_ids:
            run_values = flattened.get(run_id, {})
            value = run_values.get(key, _MISSING)
            normalized.append(_normalize_diff_value(value, run_dirs.get(run_id)))
            display.append(_format_diff_value(value, run_dirs.get(run_id)))
        if normalized and all(item == normalized[0] for item in normalized):
            continue
        rows.append([key, *display])

    if not rows:
        return None

    headers = ["key", *ordered_ids]
    return _render_table(headers, rows)


def _flatten_config_for_diff(
    payload: Mapping[str, Any], prefix: str | None = None
) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_config_for_diff(value, path))
        else:
            flattened[path] = value
    return flattened


def _normalize_diff_value(value: Any, run_dir: str | None) -> Any:
    if value is _MISSING:
        return _MISSING
    value = _normalise_run_path(value, run_dir)
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return value


def _format_diff_value(value: Any, run_dir: str | None) -> str:
    if value is _MISSING:
        return "-"
    value = _normalise_run_path(value, run_dir)
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _normalise_run_path(value: Any, run_dir: str | None) -> Any:
    if not run_dir or not isinstance(value, str):
        return value
    run_dir = str(run_dir).rstrip("/\\")
    candidate = value.replace("\\", "/")
    normalized_root = run_dir.replace("\\", "/")
    if candidate == normalized_root:
        return "<run_root>"
    if candidate.startswith(f"{normalized_root}/"):
        suffix = candidate[len(normalized_root) + 1 :]
        return f"<run_root>/{suffix}" if suffix else "<run_root>"
    return value


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
    stripped = value.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    lowered = stripped.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    return stripped


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
    if getattr(args, "command", None) == "visualize":
        _handle_visualize_command(args)
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
