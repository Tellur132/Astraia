"""Command line interface for the Anemoi MVP skeleton."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import yaml

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
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise SystemExit("Configuration root must be a mapping (YAML dictionary).")

    return data


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate the presence of required fields for the MVP optimization loop."""

    def require(path: List[str]) -> str | None:
        current: Any = config
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return ".".join(path)
            current = current[key]
        return None

    required_paths = [
        ["metadata", "name"],
        ["metadata", "description"],
        ["search", "library"],
        ["search", "direction"],
        ["search", "metric"],
        ["search", "n_trials"],
        ["stopping", "max_trials"],
        ["report", "metrics"],
        ["search_space"],
        ["evaluator", "module"],
        ["evaluator", "callable"],
    ]

    missing = [path for path in (require(p) for p in required_paths) if path is not None]
    return missing


def summarize_config(config: Dict[str, Any]) -> str:
    metadata = config.get("metadata", {})
    search = config.get("search", {})
    stopping = config.get("stopping", {})
    report = config.get("report", {})

    lines = [
        f"Experiment name : {metadata.get('name', 'N/A')}",
        f"Description    : {metadata.get('description', 'N/A')}",
        "",
        "[Search]",
        f"  Library      : {search.get('library', 'N/A')}",
        f"  Sampler      : {search.get('sampler', 'N/A')}",
        f"  Trials       : {search.get('n_trials', 'N/A')}",
        f"  Direction    : {search.get('direction', 'N/A')}",
        f"  Metric       : {search.get('metric', 'kl')}",
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
    if result.early_stopped_reason:
        lines.append(f"Early stop      : {result.early_stopped_reason}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    missing = validate_config(config)
    if missing:
        formatted = ", ".join(missing)
        raise SystemExit(
            "Configuration is missing required fields for the MVP step: " + formatted
        )

    if args.as_json:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return

    if args.summarize:
        print(summarize_config(config))
        return

    from .optimization import run_optimization

    result = run_optimization(config)
    print(format_result(result))


if __name__ == "__main__":
    main()
