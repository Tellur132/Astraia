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
    direction: str
    metric: str
    study_name: str | None = None

    @model_validator(mode="after")
    def validate_search(self) -> "SearchConfig":
        self.library = self.library.lower().strip()
        if self.library != "optuna":
            raise ValueError("search.library must be 'optuna'")

        self.sampler = self.sampler.lower().strip()
        if self.sampler not in {"tpe", "random"}:
            raise ValueError("search.sampler must be 'tpe' or 'random'")

        self.direction = self.direction.lower().strip()
        if self.direction not in {"minimize", "maximize"}:
            raise ValueError("search.direction must be 'minimize' or 'maximize'")

        if self.n_trials <= 0:
            raise ValueError("search.n_trials must be a positive integer")

        if not self.metric.strip():
            raise ValueError("search.metric must be a non-empty string")

        return self


class StoppingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_trials: int
    max_time_minutes: float | None = None
    no_improve_patience: int | None = None

    @model_validator(mode="after")
    def validate_numbers(self) -> "StoppingConfig":
        if self.max_trials <= 0:
            raise ValueError("stopping.max_trials must be positive")
        if self.max_time_minutes is not None and self.max_time_minutes <= 0:
            raise ValueError("stopping.max_time_minutes must be positive when provided")
        if self.no_improve_patience is not None and self.no_improve_patience <= 0:
            raise ValueError("stopping.no_improve_patience must be positive when provided")
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

    @model_validator(mode="after")
    def validate_strings(self) -> "LLMConfig":
        self.provider = self.provider.lower().strip()
        if not self.provider:
            raise ValueError("llm.provider must be a non-empty string")

        if not self.model.strip():
            raise ValueError("llm.model must be a non-empty string")

        if self.usage_log is not None and not self.usage_log.strip():
            raise ValueError("llm.usage_log must be a non-empty string when provided")

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


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: MetadataConfig
    seed: int | None = None
    search: SearchConfig
    stopping: StoppingConfig
    planner: PlannerConfig | None = None
    llm: LLMConfig | None = None
    llm_guidance: LLMGuidanceConfig | None = None
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

        primary_metric = self.search.metric.lower()
        metric_names = {metric.lower() for metric in self.report.metrics}
        if primary_metric not in metric_names:
            raise ValueError("search.metric must be included in report.metrics")

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

        return self


__all__ = ["OptimizationConfig", "ValidationError", "LLMConfig", "LLMGuidanceConfig"]
