"""Core optimization loop implementation for the Anemoi MVP."""
from __future__ import annotations

import csv
import importlib
import time
from dataclasses import dataclass
from pathlib import Path
import inspect
from typing import Any, Callable, Dict, Iterable, List, Mapping

import optuna

from .evaluators import BaseEvaluator


@dataclass
class OptimizationResult:
    """Container for summarising the optimization run."""

    trials_completed: int
    best_params: Dict[str, Any]
    best_metrics: Dict[str, float]
    best_value: float
    early_stopped_reason: str | None = None


def run_optimization(config: Mapping[str, Any]) -> OptimizationResult:
    """Execute the optimization loop using the provided configuration."""

    ensure_directories(config)
    evaluator = load_evaluator(config["evaluator"])
    library = str(config["search"].get("library", "")).lower()
    if library != "optuna":
        raise ValueError(f"Unsupported search library: {library or 'unknown'}")
    sampler = build_sampler(config["search"], config.get("seed"))
    study = create_study(config["search"], sampler)

    search_space = config["search_space"]
    metric_name = config["search"]["metric"]
    max_trials = int(config["stopping"].get("max_trials", config["search"]["n_trials"]))
    max_time_minutes = config["stopping"].get("max_time_minutes")
    patience = config["stopping"].get("no_improve_patience")
    seed = config.get("seed")

    log_file = Path(config.get("artifacts", {}).get("log_file", "runs/log.csv"))

    trials_completed = 0
    early_stop_reason: str | None = None
    best_value: float | None = None
    best_metrics: Dict[str, float] = {}
    best_params: Dict[str, Any] = {}

    with TrialLogger(
        log_file,
        search_space.keys(),
        config["report"]["metrics"],
    ) as writer:
        start_ts = time.time()
        no_improve_counter = 0

        for _ in range(max_trials):
            if max_time_minutes is not None:
                elapsed = (time.time() - start_ts) / 60
                if elapsed >= float(max_time_minutes):
                    early_stop_reason = "max_time_minutes reached"
                    break

            trial = study.ask()
            params = sample_params(trial, search_space)
            metrics = evaluator(params, seed)

            if metric_name not in metrics:
                raise RuntimeError(
                    f"Evaluator did not return primary metric '{metric_name}'."
                )

            primary_value = float(metrics[metric_name])
            study.tell(trial, primary_value)
            trials_completed += 1

            writer.log(trial.number, params, metrics)

            improved = update_best(
                study.direction,
                primary_value,
                best_value,
            )
            if improved:
                best_value = primary_value
                best_metrics = dict(metrics)
                best_params = dict(params)
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            if patience is not None and no_improve_counter >= int(patience):
                early_stop_reason = "no_improve_patience reached"
                break

            if trials_completed >= int(config["search"].get("n_trials", max_trials)):
                break

    if trials_completed == 0:
        raise RuntimeError("Optimization did not record any trials.")

    if best_value is None:
        best_value = float(study.best_value)

    build_report(
        config,
        best_params,
        best_metrics,
        trials_completed,
        early_stop_reason,
    )

    return OptimizationResult(
        trials_completed=trials_completed,
        best_params=best_params,
        best_metrics=best_metrics,
        best_value=best_value,
        early_stopped_reason=early_stop_reason,
    )


def ensure_directories(config: Mapping[str, Any]) -> None:
    artifacts = config.get("artifacts", {})
    report = config.get("report", {})

    log_path = Path(artifacts.get("log_file", "runs/log.csv"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    report_dir = Path(report.get("output_dir", "reports"))
    report_dir.mkdir(parents=True, exist_ok=True)

    run_root = artifacts.get("run_root")
    if run_root:
        Path(run_root).mkdir(parents=True, exist_ok=True)


def load_evaluator(config: Mapping[str, Any]) -> Callable[[Dict[str, Any], int | None], Dict[str, float]]:
    module = importlib.import_module(config["module"])
    target = getattr(module, config["callable"])

    evaluator_obj: Any
    if isinstance(target, BaseEvaluator):
        evaluator_obj = target
    elif isinstance(target, type) and issubclass(target, BaseEvaluator):
        evaluator_obj = target()  # type: ignore[call-arg]
    elif callable(target):
        signature = inspect.signature(target)
        if len(signature.parameters) <= 1:
            evaluator_obj = target(config)
        else:
            evaluator_obj = target
    else:
        raise TypeError(
            "Evaluator callable must be a function, factory, or BaseEvaluator instance."
        )

    if isinstance(evaluator_obj, BaseEvaluator):
        return evaluator_obj.evaluate
    if callable(evaluator_obj):
        return evaluator_obj
    raise TypeError("Evaluator factory did not return a callable or BaseEvaluator instance.")


def build_sampler(search_cfg: Mapping[str, Any], seed: int | None) -> optuna.samplers.BaseSampler:
    sampler_name = search_cfg.get("sampler", "tpe").lower()
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"Unsupported sampler: {sampler_name}")


def create_study(search_cfg: Mapping[str, Any], sampler: optuna.samplers.BaseSampler) -> optuna.study.Study:
    direction = search_cfg.get("direction", "minimize").lower()
    if direction not in {"minimize", "maximize"}:
        raise ValueError(f"Unsupported direction: {direction}")

    return optuna.create_study(
        study_name=search_cfg.get("study_name"),
        direction=direction,
        sampler=sampler,
    )


def sample_params(trial: optuna.trial.Trial, space: Mapping[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, spec in space.items():
        param_type = spec.get("type")
        if param_type == "float":
            step = spec.get("step")
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
                step=None if step is None else float(step),
            )
        elif param_type == "int":
            step = spec.get("step")
            int_step = int(step) if step is not None else 1
            params[name] = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int_step,
            )
        elif param_type == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ValueError(f"Unsupported parameter type for '{name}': {param_type}")
    return params


def update_best(
    direction: optuna.study.StudyDirection,
    current_value: float,
    best_value: float | None,
) -> bool:
    if best_value is None:
        return True
    if direction == optuna.study.StudyDirection.MINIMIZE:
        return current_value < best_value
    return current_value > best_value


class TrialLogger:
    """Utility to append trial information to a CSV log file."""

    def __init__(
        self,
        path: Path,
        param_names: Iterable[str],
        metric_names: Iterable[str],
    ) -> None:
        self.path = path
        self.param_names = list(param_names)
        self.metric_names = list(metric_names)

        write_header = not path.exists()
        self._fh = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._fh,
            fieldnames=[
                "trial",
                *[f"param_{name}" for name in self.param_names],
                *[f"metric_{name}" for name in self.metric_names],
            ],
        )
        if write_header:
            self._writer.writeheader()

    def log(self, trial_number: int, params: Mapping[str, Any], metrics: Mapping[str, float]) -> None:
        row: Dict[str, Any] = {"trial": trial_number}
        for name in self.param_names:
            row[f"param_{name}"] = params.get(name)
        for name in self.metric_names:
            row[f"metric_{name}"] = metrics.get(name)
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "TrialLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001 - guard for interpreter shutdown
            pass


def build_report(
    config: Mapping[str, Any],
    best_params: Mapping[str, Any],
    best_metrics: Mapping[str, float],
    trials_completed: int,
    early_stop_reason: str | None,
) -> Path:
    report_cfg = config.get("report", {})
    report_dir = Path(report_cfg.get("output_dir", "reports"))
    filename = report_cfg.get("filename") or f"{config['metadata']['name']}.md"
    report_path = report_dir / filename

    lines: List[str] = [
        f"# Experiment Report — {config['metadata']['name']}",
        "",
        f"Description: {config['metadata'].get('description', '')}",
        "",
        f"Trials executed: {trials_completed}",
        f"Primary metric: {config['search']['metric']}",
        "",
        "## Best Parameters",
    ]
    lines.extend([f"- **{name}**: {value}" for name, value in best_params.items()])
    lines.append("")
    lines.append("## Best Metrics")
    lines.extend([f"- **{name}**: {value}" for name, value in best_metrics.items()])
    if early_stop_reason:
        lines.extend(["", f"_Early stopping_: {early_stop_reason}"])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
