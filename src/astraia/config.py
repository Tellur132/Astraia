"""Configuration schema and validation using a lightweight Pydantic shim."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

try:  # pragma: no cover - use real pydantic when available
    from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
except ImportError:  # pragma: no cover - offline fallback
    from ._compat.pydantic import (  # type: ignore[assignment]
        BaseModel,
        ConfigDict,
        ValidationError,
        model_validator,
    )


class MetadataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str

    @model_validator(mode="after")
    def validate_strings(self) -> "MetadataConfig":
        if not self.name.strip():
            raise ValueError("metadata.name must be a non-empty string")
        if not self.description.strip():
            raise ValueError("metadata.description must be a non-empty string")
        return self


class SearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    library: str = "optuna"
    sampler: str = "tpe"
    n_trials: int
    direction: str | List[str]
    metric: str | List[str]
    study_name: str | None = None

    @model_validator(mode="after")
    def validate_search(self) -> "SearchConfig":
        self.library = self.library.lower().strip()
        if self.library != "optuna":
            raise ValueError("search.library must be 'optuna'")

        self.sampler = self.sampler.lower().strip()
        allowed_samplers = {
            "tpe",
            "random",
            "nsga2",
            "motpe",
            "nsgaiii",
            "moead",
            "mocma",
        }
        if self.sampler not in allowed_samplers:
            raise ValueError(
                "search.sampler must be one of " + ", ".join(sorted(allowed_samplers))
            )

        directions = self._normalise_sequence(self.direction, field="direction")
        normalised_dirs: List[str] = []
        for direction in directions:
            direction_lc = direction.lower().strip()
            if direction_lc not in {"minimize", "maximize"}:
                raise ValueError("search.direction entries must be 'minimize' or 'maximize'")
            normalised_dirs.append(direction_lc)
        self.direction = (
            normalised_dirs[0] if len(normalised_dirs) == 1 else normalised_dirs
        )

        if self.n_trials <= 0:
            raise ValueError("search.n_trials must be a positive integer")

        metrics = self._normalise_sequence(self.metric, field="metric")
        normalised_metrics: List[str] = []
        for metric in metrics:
            metric_name = metric.strip()
            if not metric_name:
                raise ValueError("search.metric entries must be non-empty strings")
            normalised_metrics.append(metric_name)
        self.metric = normalised_metrics[0] if len(normalised_metrics) == 1 else normalised_metrics

        if len(self.metric_names) != len(self.direction_names):
            raise ValueError(
                "search.metric and search.direction must have the same number of entries"
            )

        return self

    @property
    def metric_names(self) -> List[str]:
        if isinstance(self.metric, str):
            return [self.metric]
        return list(self.metric)

    @property
    def direction_names(self) -> List[str]:
        if isinstance(self.direction, str):
            return [self.direction]
        return list(self.direction)

    @staticmethod
    def _normalise_sequence(value: str | List[str], *, field: str) -> List[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            if not value:
                raise ValueError(f"search.{field} must not be empty")
            return list(value)
        raise TypeError(f"search.{field} must be a string or list of strings")


class StoppingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_trials: int
    max_time_minutes: float | None = None
    no_improve_patience: int | None = None
    cost_metric: str | None = None
    max_total_cost: float | None = None

    @model_validator(mode="after")
    def validate_numbers(self) -> "StoppingConfig":
        if self.max_trials <= 0:
            raise ValueError("stopping.max_trials must be positive")
        if self.max_time_minutes is not None and self.max_time_minutes <= 0:
            raise ValueError("stopping.max_time_minutes must be positive when provided")
        if self.no_improve_patience is not None and self.no_improve_patience <= 0:
            raise ValueError("stopping.no_improve_patience must be positive when provided")
        if self.cost_metric is not None:
            metric = self.cost_metric.strip()
            if not metric:
                raise ValueError("stopping.cost_metric must be a non-empty string when provided")
            self.cost_metric = metric
        if self.max_total_cost is not None and self.max_total_cost <= 0:
            raise ValueError("stopping.max_total_cost must be positive when provided")
        if self.max_total_cost is not None and self.cost_metric is None:
            raise ValueError(
                "stopping.cost_metric must be specified when max_total_cost is provided"
            )
        return self


class PlannerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = "rule"
    enabled: bool = False
    role: str | None = None
    prompt_template: str | None = None
    config_path: str | None = None

    @model_validator(mode="after")
    def validate_strings(self) -> "PlannerConfig":
        self.backend = self.backend.lower().strip()
        if self.backend not in {"rule", "llm"}:
            raise ValueError("planner.backend must be 'rule' or 'llm'")

        if self.config_path is not None and not self.config_path.strip():
            raise ValueError("planner.config_path must be a non-empty string when provided")

        if self.backend == "llm":
            if not (self.role and self.role.strip()):
                raise ValueError("planner.role must be a non-empty string for llm backend")
            if not (self.prompt_template and self.prompt_template.strip()):
                raise ValueError(
                    "planner.prompt_template must be a non-empty string for llm backend"
                )
        else:
            if self.role is not None and not self.role.strip():
                raise ValueError("planner.role must be a non-empty string when provided")
            if self.prompt_template is not None and not self.prompt_template.strip():
                raise ValueError(
                    "planner.prompt_template must be a non-empty string when provided"
                )

        return self


class ReportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = "reports"
    filename: str | None = None
    metrics: List[str]

    @model_validator(mode="after")
    def validate_metrics(self) -> "ReportConfig":
        if not self.metrics:
            raise ValueError("report.metrics must contain at least one metric name")
        lower_seen: set[str] = set()
        for metric in self.metrics:
            if not metric.strip():
                raise ValueError("report.metrics entries must be non-empty strings")
            metric_lc = metric.lower()
            if metric_lc in lower_seen:
                raise ValueError("report.metrics entries must be unique (case-insensitive)")
            lower_seen.add(metric_lc)
        if self.output_dir and not self.output_dir.strip():
            raise ValueError("report.output_dir must be a non-empty string")
        if self.filename is not None and not self.filename.strip():
            raise ValueError("report.filename must be a non-empty string when provided")
        return self


class ArtifactsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_root: str | None = None
    log_file: str = "runs/log.csv"

    @model_validator(mode="after")
    def validate_paths(self) -> "ArtifactsConfig":
        if self.run_root is not None and not self.run_root.strip():
            raise ValueError("artifacts.run_root must be a non-empty string when provided")
        if not self.log_file.strip():
            raise ValueError("artifacts.log_file must be a non-empty string")
        return self


class EvaluatorConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    module: str
    callable: str

    @model_validator(mode="after")
    def validate_strings(self) -> "EvaluatorConfig":
        if not self.module.strip():
            raise ValueError("evaluator.module must be a non-empty string")
        if not self.callable.strip():
            raise ValueError("evaluator.callable must be a non-empty string")
        return self


class FloatParam(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    low: float
    high: float
    step: float | None = None
    log: bool = False

    @model_validator(mode="after")
    def validate_float(self) -> "FloatParam":
        if self.type.lower() != "float":
            raise ValueError("Search space entry type must be 'float'")
        if self.low >= self.high:
            raise ValueError("float parameter requires low < high")
        if self.step is not None and self.step <= 0:
            raise ValueError("float parameter step must be positive when provided")
        return self


class IntParam(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    low: int
    high: int
    step: int | None = None
    log: bool | None = None

    @model_validator(mode="after")
    def validate_int(self) -> "IntParam":
        if self.type.lower() != "int":
            raise ValueError("Search space entry type must be 'int'")
        if self.low >= self.high:
            raise ValueError("int parameter requires low < high")
        if self.step is not None and self.step <= 0:
            raise ValueError("int parameter step must be positive when provided")
        return self


class CategoricalParam(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    choices: List[Any]

    @model_validator(mode="after")
    def validate_choices(self) -> "CategoricalParam":
        if self.type.lower() != "categorical":
            raise ValueError("Search space entry type must be 'categorical'")
        if not self.choices:
            raise ValueError("categorical parameter requires at least one choice")
        return self


PARAMETER_MODELS = {
    "float": FloatParam,
    "int": IntParam,
    "categorical": CategoricalParam,
}


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    usage_log: str | None = None
    max_calls: int | None = None
    max_tokens_per_run: int | None = None
    budget_usd: float | None = None
    prompt_cost_per_1k: float | None = None
    completion_cost_per_1k: float | None = None

    @model_validator(mode="after")
    def validate_strings(self) -> "LLMConfig":
        self.provider = self.provider.lower().strip()
        if not self.provider:
            raise ValueError("llm.provider must be a non-empty string")

        if not self.model.strip():
            raise ValueError("llm.model must be a non-empty string")

        if self.usage_log is not None and not self.usage_log.strip():
            raise ValueError("llm.usage_log must be a non-empty string when provided")

        if self.max_calls is not None and self.max_calls <= 0:
            raise ValueError("llm.max_calls must be positive when provided")
        if self.max_tokens_per_run is not None and self.max_tokens_per_run <= 0:
            raise ValueError("llm.max_tokens_per_run must be positive when provided")
        if self.budget_usd is not None and self.budget_usd <= 0:
            raise ValueError("llm.budget_usd must be positive when provided")
        if self.prompt_cost_per_1k is not None and self.prompt_cost_per_1k <= 0:
            raise ValueError("llm.prompt_cost_per_1k must be positive when provided")
        if (
            self.completion_cost_per_1k is not None
            and self.completion_cost_per_1k <= 0
        ):
            raise ValueError(
                "llm.completion_cost_per_1k must be positive when provided"
            )
        if self.budget_usd is not None:
            if self.prompt_cost_per_1k is None or self.completion_cost_per_1k is None:
                raise ValueError(
                    "llm.prompt_cost_per_1k and llm.completion_cost_per_1k are required when budget_usd is set"
                )

        return self


class LLMGuidanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    problem_summary: str | None = None
    objective: str | None = None
    n_proposals: int = 1
    max_retries: int = 2
    base_temperature: float = 0.7
    min_temperature: float = 0.1

    @model_validator(mode="after")
    def validate_fields(self) -> "LLMGuidanceConfig":
        if self.enabled:
            if not (self.problem_summary and self.problem_summary.strip()):
                raise ValueError(
                    "llm_guidance.problem_summary must be a non-empty string when enabled"
                )
            self.problem_summary = self.problem_summary.strip()
            if not (self.objective and self.objective.strip()):
                raise ValueError(
                    "llm_guidance.objective must be a non-empty string when enabled"
                )
            self.objective = self.objective.strip()
        if self.problem_summary is not None:
            self.problem_summary = self.problem_summary.strip()
        if self.objective is not None:
            self.objective = self.objective.strip()
        if self.n_proposals <= 0:
            raise ValueError("llm_guidance.n_proposals must be positive")
        if self.max_retries < 0:
            raise ValueError("llm_guidance.max_retries must be non-negative")
        if self.base_temperature <= 0:
            raise ValueError("llm_guidance.base_temperature must be positive")
        if self.min_temperature <= 0:
            raise ValueError("llm_guidance.min_temperature must be positive")
        if self.min_temperature > self.base_temperature:
            raise ValueError(
                "llm_guidance.min_temperature must be less than or equal to base_temperature"
            )
        return self


class MetaSearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    interval: int = 10
    summary_trials: int = 10

    @model_validator(mode="after")
    def validate_fields(self) -> "MetaSearchConfig":
        if self.interval <= 0:
            raise ValueError("meta_search.interval must be a positive integer")
        if self.summary_trials <= 0:
            raise ValueError("meta_search.summary_trials must be a positive integer")
        return self


class LLMCriticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    max_history: int = 200
    prompt_preamble: str | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "LLMCriticConfig":
        if self.max_history <= 0:
            raise ValueError("llm_critic.max_history must be a positive integer")
        if self.prompt_preamble is not None:
            text = self.prompt_preamble.strip()
            if not text:
                raise ValueError(
                    "llm_critic.prompt_preamble must be a non-empty string when provided"
                )
            self.prompt_preamble = text
        return self


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: MetadataConfig
    seed: int | None = None
    search: SearchConfig
    stopping: StoppingConfig
    planner: PlannerConfig | None = None
    llm: LLMConfig | None = None
    llm_guidance: LLMGuidanceConfig | None = None
    meta_search: MetaSearchConfig | None = None
    llm_critic: LLMCriticConfig | None = None
    search_space: Dict[str, Dict[str, Any]]
    evaluator: EvaluatorConfig
    report: ReportConfig
    artifacts: ArtifactsConfig | None = None

    @model_validator(mode="after")
    def validate_all(self) -> "OptimizationConfig":
        if not self.search_space:
            raise ValueError("search_space must define at least one parameter")

        normalised_space: Dict[str, Dict[str, Any]] = {}
        for name, spec in self.search_space.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("search_space parameter names must be non-empty strings")
            if not isinstance(spec, Mapping):
                raise ValueError(f"search_space.{name} must be a mapping")
            param_type = str(spec.get("type", "")).lower()
            model_cls = PARAMETER_MODELS.get(param_type)
            if model_cls is None:
                raise ValueError(f"search_space.{name}.type '{param_type}' is not supported")
            normalised_space[name] = model_cls.model_validate(dict(spec)).model_dump()

        self.search_space = normalised_space

        metric_names = {metric.lower() for metric in self.report.metrics}
        search_metrics = [metric.lower() for metric in self.search.metric_names]
        missing = [metric for metric in search_metrics if metric not in metric_names]
        if missing:
            raise ValueError("search.metric entries must be included in report.metrics")

        if self.llm is not None and not self.llm.usage_log:
            run_root = None
            if self.artifacts is not None:
                artifacts = self.artifacts.model_dump()
                run_root = artifacts.get("run_root")
            if run_root:
                usage_path = Path(run_root) / "llm_usage.csv"
                self.llm.usage_log = str(usage_path)

        if self.llm_guidance is not None and self.llm_guidance.enabled and self.llm is None:
            raise ValueError("llm_guidance requires llm configuration when enabled")

        if self.llm_critic is not None and self.llm_critic.enabled and self.llm is None:
            raise ValueError("llm_critic requires llm configuration when enabled")

        return self


__all__ = [
    "OptimizationConfig",
    "ValidationError",
    "LLMConfig",
    "LLMGuidanceConfig",
    "MetaSearchConfig",
    "LLMCriticConfig",
]
