"""Configuration schema and validation using a lightweight Pydantic shim."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

try:  # pragma: no cover - use real pydantic when available
    from pydantic import (
        BaseModel,
        ConfigDict,
        ValidationError,
        field_validator,
        model_validator,
    )
except ImportError:  # pragma: no cover - offline fallback
    from ._compat.pydantic import (  # type: ignore[assignment]
        BaseModel,
        ConfigDict,
        ValidationError,
        field_validator,
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
    direction: str | List[str] | Mapping[str, Any] | None = "minimize"
    metric: str | List[str] | Mapping[str, Any] | List[Mapping[str, Any]] | None = None
    metrics: List[str | Mapping[str, Any]] | None = None
    directions: List[str] | None = None
    multi_objective: bool = False
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
            "nevergrad",
            "de",
        }
        if self.sampler not in allowed_samplers:
            raise ValueError(
                "search.sampler must be one of " + ", ".join(sorted(allowed_samplers))
            )

        self.multi_objective = bool(self.multi_objective)

        metrics, directions = self._resolve_metrics_and_directions()
        if not metrics:
            raise ValueError("search.metric must define at least one objective")

        if not self.multi_objective and len(metrics) > 1:
            raise ValueError(
                "search.multi_objective must be true when multiple metrics are defined"
            )
        if self.multi_objective and len(metrics) < 2:
            raise ValueError(
                "search.metrics must define at least two objectives when multi_objective is true"
            )

        if self.multi_objective and len(directions) != len(metrics):
            raise ValueError(
                "search.directions must have the same number of entries as search.metrics"
            )

        if self.n_trials <= 0:
            raise ValueError("search.n_trials must be a positive integer")

        if self.multi_objective:
            self.metric = None
            self.direction = None
        else:
            self.metric = metrics[0]
            self.direction = directions[0]
        self.metrics = metrics
        self.directions = directions

        return self

    @property
    def metric_names(self) -> List[str]:
        if self.metrics:
            return list(self.metrics)
        if isinstance(self.metric, str):
            return [self.metric]
        if isinstance(self.metric, list):
            return list(self.metric)
        return []

    @property
    def direction_names(self) -> List[str]:
        if self.directions:
            return list(self.directions)
        if isinstance(self.direction, str):
            return [self.direction]
        if isinstance(self.direction, list):
            return list(self.direction)
        return []

    def _resolve_metrics_and_directions(self) -> tuple[List[str], List[str]]:
        metrics_source: Any = self.metrics if self.metrics is not None else self.metric
        if metrics_source is None:
            raise ValueError("search.metric must be provided")

        directions_source: Any = self.directions if self.directions is not None else self.direction
        if directions_source is None:
            directions_source = "minimize"

        entries: List[Any]
        if isinstance(metrics_source, list):
            if not metrics_source:
                raise ValueError("search.metrics must not be empty")
            entries = list(metrics_source)
        else:
            entries = [metrics_source]

        metrics: List[str] = []
        directions: List[str] = []
        for idx, entry in enumerate(entries):
            metric_name, explicit_direction = self._parse_metric_entry(entry, idx)
            metrics.append(metric_name)
            directions.append(
                self._normalise_direction(
                    explicit_direction,
                    default_source=directions_source,
                    idx=idx,
                    total=len(entries),
                )
            )
        return metrics, directions

    def _parse_metric_entry(self, entry: Any, idx: int) -> tuple[str, str | None]:
        if isinstance(entry, Mapping):
            metric_name = str(entry.get("name", "")).strip()
            if not metric_name:
                raise ValueError("search.metrics entries must include a non-empty 'name'")
            direction = entry.get("direction")
            return metric_name, str(direction) if direction is not None else None

        metric_name = str(entry).strip()
        if not metric_name:
            raise ValueError("search.metric entries must be non-empty strings")
        return metric_name, None

    def _normalise_direction(
        self,
        explicit_direction: str | None,
        *,
        default_source: Any,
        idx: int,
        total: int,
    ) -> str:
        direction_value: Any
        if explicit_direction is not None:
            direction_value = explicit_direction
        elif isinstance(default_source, list):
            if not default_source:
                raise ValueError("search.directions must not be empty when provided")
            if len(default_source) == 1:
                direction_value = default_source[0]
            elif idx < len(default_source):
                direction_value = default_source[idx]
            else:
                raise ValueError("search.directions must match the number of metrics")
        else:
            direction_value = default_source

        direction = str(direction_value).lower().strip()
        if direction not in {"minimize", "maximize"}:
            raise ValueError("search.direction entries must be 'minimize' or 'maximize'")
        return direction


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


class LLMOnlyParam(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    default: Any | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "LLMOnlyParam":
        if self.type.lower() != "llm_only":
            raise ValueError("Search space entry type must be 'llm_only'")
        if self.default is not None and isinstance(self.default, str):
            cleaned = self.default.strip()
            self.default = cleaned
        return self


PARAMETER_MODELS = {
    "float": FloatParam,
    "int": IntParam,
    "categorical": CategoricalParam,
    "llm_only": LLMOnlyParam,
}


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    usage_log: str | None = None
    trace_log: str | None = None
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
        if self.trace_log is not None and not self.trace_log.strip():
            raise ValueError("llm.trace_log must be a non-empty string when provided")

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


class PlannerRoleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_template: str | None = None
    llm: LLMConfig | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "PlannerRoleConfig":
        if self.prompt_template is not None:
            template = self.prompt_template.strip()
            if not template:
                raise ValueError(
                    "planner.roles entries must define a non-empty prompt_template when provided"
                )
            self.prompt_template = template
        return self


class PlannerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = "rule"
    enabled: bool = False
    role: str | None = None
    prompt_template: str | None = None
    config_path: str | None = None
    roles: Dict[str, PlannerRoleConfig] | None = None

    @model_validator(mode="after")
    def validate_strings(self) -> "PlannerConfig":
        self.backend = self.backend.lower().strip()
        if self.backend not in {"rule", "llm"}:
            raise ValueError("planner.backend must be 'rule' or 'llm'")

        if self.config_path is not None and not self.config_path.strip():
            raise ValueError("planner.config_path must be a non-empty string when provided")

        if self.roles is not None:
            cleaned: Dict[str, PlannerRoleConfig] = {}
            for name, cfg in self.roles.items():
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("planner.roles keys must be non-empty strings")
                cleaned[name.strip()] = cfg
            self.roles = cleaned

        if self.backend == "llm":
            has_named_roles = bool(self.roles)
            has_primary_role = bool(self.role and self.role.strip())
            if not (has_named_roles or has_primary_role):
                raise ValueError(
                    "planner.role or planner.roles must be provided for llm backend"
                )
            if has_primary_role:
                role_name = self.role.strip()
                self.role = role_name
                if not (self.prompt_template and self.prompt_template.strip()):
                    raise ValueError(
                        "planner.prompt_template must be a non-empty string for llm backend"
                    )
                self.prompt_template = self.prompt_template.strip()
        else:
            if self.role is not None and not self.role.strip():
                raise ValueError("planner.role must be a non-empty string when provided")
            if self.prompt_template is not None and not self.prompt_template.strip():
                raise ValueError(
                    "planner.prompt_template must be a non-empty string when provided"
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
    policies: List["MetaPolicyConfig"] | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "MetaSearchConfig":
        if self.interval <= 0:
            raise ValueError("meta_search.interval must be a positive integer")
        if self.summary_trials <= 0:
            raise ValueError("meta_search.summary_trials must be a positive integer")
        return self


class PolicyMetricThreshold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str
    value: float

    @model_validator(mode="after")
    def validate_fields(self) -> "PolicyMetricThreshold":
        metric_name = self.metric.strip()
        if not metric_name:
            raise ValueError(
                "meta_search.policies.metric must be a non-empty string when provided"
            )
        self.metric = metric_name
        return self


class PolicyConditionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    no_improve: int | None = None
    metric_below: PolicyMetricThreshold | None = None
    metric_above: PolicyMetricThreshold | None = None
    sampler: List[str] | None = None
    min_trials: int | None = None

    @field_validator("sampler", mode="before")
    @classmethod
    def normalise_sampler(cls, value: Any) -> List[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            normalised: List[str] = []
            for item in value:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(
                        "meta_search.policies.when.sampler entries must be non-empty strings"
                    )
                normalised.append(item.strip().lower())
            return normalised
        raise TypeError(
            "meta_search.policies.when.sampler must be a string or list of strings"
        )

    @model_validator(mode="after")
    def validate_fields(self) -> "PolicyConditionConfig":
        if self.no_improve is not None and self.no_improve <= 0:
            raise ValueError("meta_search.policies.when.no_improve must be positive")
        if self.min_trials is not None and self.min_trials < 0:
            raise ValueError("meta_search.policies.when.min_trials must be non-negative")
        if self.sampler is not None and not self.sampler:
            raise ValueError("meta_search.policies.when.sampler must contain entries")
        if not any(
            [
                self.no_improve is not None,
                self.metric_below is not None,
                self.metric_above is not None,
                self.sampler,
                self.min_trials is not None,
            ]
        ):
            raise ValueError("meta_search.policies.when must define at least one condition")
        return self


class PolicyActionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sampler: str | None = None
    rescale: Dict[str, float] | None = None
    trial_budget: int | None = None
    max_trials: int | None = None
    patience: int | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "PolicyActionConfig":
        if self.sampler is not None:
            sampler_name = self.sampler.strip().lower()
            if not sampler_name:
                raise ValueError(
                    "meta_search.policies.then.sampler must be a non-empty string when provided"
                )
            self.sampler = sampler_name
        if self.rescale is not None:
            cleaned: Dict[str, float] = {}
            for name, value in self.rescale.items():
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(
                        "meta_search.policies.then.rescale keys must be non-empty strings"
                    )
                factor = float(value)
                if not (0.05 <= factor <= 1.0):
                    raise ValueError(
                        "meta_search.policies.then.rescale values must be between 0.05 and 1.0"
                    )
                cleaned[name.strip()] = factor
            self.rescale = cleaned
        if self.trial_budget is not None and self.trial_budget <= 0:
            raise ValueError(
                "meta_search.policies.then.trial_budget must be positive when provided"
            )
        if self.max_trials is not None and self.max_trials <= 0:
            raise ValueError(
                "meta_search.policies.then.max_trials must be positive when provided"
            )
        if self.patience is not None and self.patience < 0:
            raise ValueError(
                "meta_search.policies.then.patience must be non-negative when provided"
            )
        if self.notes is not None:
            note = self.notes.strip()
            if not note:
                raise ValueError(
                    "meta_search.policies.then.notes must be a non-empty string when provided"
                )
            self.notes = note
        if not any(
            [
                self.sampler is not None,
                self.rescale,
                self.trial_budget is not None,
                self.max_trials is not None,
                self.patience is not None,
                self.notes is not None,
            ]
        ):
            raise ValueError(
                "meta_search.policies.then must define at least one adjustment directive"
            )
        return self


class MetaPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    when: PolicyConditionConfig
    then: PolicyActionConfig
    cooldown_trials: int = 0
    trigger_once: bool = False

    @model_validator(mode="after")
    def validate_fields(self) -> "MetaPolicyConfig":
        if self.name is not None:
            name = self.name.strip()
            if not name:
                raise ValueError(
                    "meta_search.policies.name must be a non-empty string when provided"
                )
            self.name = name
        if self.cooldown_trials < 0:
            raise ValueError(
                "meta_search.policies.cooldown_trials must be non-negative"
            )
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

        candidate_override_available = False
        if self.planner is not None and self.planner.roles is not None:
            candidate_role = self.planner.roles.get("candidate")
            if candidate_role is not None and candidate_role.llm is not None:
                candidate_override_available = True

        if (
            self.llm_guidance is not None
            and self.llm_guidance.enabled
            and self.llm is None
            and not candidate_override_available
        ):
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
