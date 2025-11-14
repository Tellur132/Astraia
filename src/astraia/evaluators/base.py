"""Base interfaces and utilities for evaluator plugins."""
from __future__ import annotations

import concurrent.futures
import contextlib
import math
import os
import random
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


MetricValue = float | int | str | bool | None
"""Supported value types in an evaluator result payload."""


EvaluatorResult = Dict[str, MetricValue]
"""Canonical mapping type returned by evaluators."""


class GracefulNaNPolicy(str, Enum):
    """Policies for handling NaN/Inf values from evaluator payloads."""

    ERROR = "error"
    COERCE_TO_INF = "coerce_to_inf"
    MARK_FAILURE = "mark_failure"


class EvaluatorInput(BaseModel):
    """Structured representation of evaluator inputs."""

    params: Dict[str, Any]
    seed: int | None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class EvaluatorOutput(BaseModel):
    """Validation schema applied to evaluator outputs."""

    kl: float
    depth: float = Field(ge=0)
    shots: float = Field(ge=0)
    params: float = Field(ge=0)
    status: str
    timed_out: bool
    terminated_early: bool
    elapsed_seconds: float | None = Field(default=None, ge=0)
    reason: str | None = None

    model_config = ConfigDict(extra="allow")

    @field_validator("status")
    @classmethod
    def _validate_status(cls, value: str) -> str:
        if value not in BaseEvaluator.VALID_STATUSES:
            raise ValueError(f"Unsupported evaluator status: {value!r}")
        return value


@dataclass(frozen=True)
class _ExecutionConfig:
    """Execution parameters resolved for a single evaluation."""

    timeout: float | None
    max_retries: int
    nan_policy: GracefulNaNPolicy


class BaseEvaluator(ABC):
    """Common interface for all evaluator implementations.

    Evaluators receive a dictionary of trial parameters and return a dictionary of
    computed metrics. The primary metric is defined by the optimization
    configuration, but evaluators are free to emit additional diagnostics.

    Concrete subclasses must implement :meth:`_evaluate_impl` and return a mapping
    that includes the required keys ``kl``, ``depth``, ``shots``, and ``params``.
    The base class converts the mapping into a normalized :class:`EvaluatorResult`
    and fills in default values for optional control fields such as ``status`` and
    ``timed_out``.
    """

    #: Keys that must always be included in an evaluator payload.
    REQUIRED_METRICS: Sequence[str] = ("kl", "depth", "shots", "params")

    #: Accepted status labels for standardized evaluator results.
    VALID_STATUSES: Sequence[str] = ("ok", "error", "timeout")

    #: Default graceful NaN policy if not provided explicitly by subclasses.
    DEFAULT_NAN_POLICY: GracefulNaNPolicy = GracefulNaNPolicy.ERROR

    def evaluate(
        self,
        params: Mapping[str, Any],
        seed: int | None = None,
        *,
        trial_timeout_sec: float | None = None,
        max_retries: int | None = None,
        graceful_nan_policy: GracefulNaNPolicy | str | None = None,
    ) -> EvaluatorResult:
        """Compute metrics for the provided parameter set.

        Parameters
        ----------
        params:
            Mapping of parameter names to values for the trial.
        seed:
            Optional deterministic seed propagated to supported RNG backends.
        trial_timeout_sec:
            Optional wall-clock timeout applied to the evaluation call. When not
            supplied the evaluator falls back to ``self.trial_timeout_sec`` when
            defined.
        max_retries:
            Number of automatic retries permitted when the evaluator raises an
            exception. Defaults to ``self.max_retries`` or zero.
        graceful_nan_policy:
            Policy used when NaN/Inf values are detected in required metrics.
        """

        input_payload = EvaluatorInput.model_validate({"params": dict(params), "seed": seed})
        exec_config = self._resolve_execution_config(
            trial_timeout_sec=trial_timeout_sec,
            max_retries=max_retries,
            graceful_nan_policy=graceful_nan_policy,
        )

        attempts = exec_config.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            start = time.perf_counter()
            try:
                raw_payload = self._execute_trial(
                    params=input_payload.params,
                    seed=input_payload.seed,
                    timeout=exec_config.timeout,
                )
                elapsed = time.perf_counter() - start
                return self._finalize_result(
                    raw_payload,
                    params=input_payload.params,
                    elapsed=elapsed,
                    nan_policy=exec_config.nan_policy,
                )
            except TimeoutError:
                elapsed = time.perf_counter() - start
                return self._timeout_result(
                    params=input_payload.params,
                    elapsed=elapsed,
                )
            except Exception as exc:  # noqa: BLE001 - propagate through retry logic
                last_error = exc
                if attempt < attempts - 1:
                    continue
                elapsed = time.perf_counter() - start
                return self._exception_result(
                    params=input_payload.params,
                    elapsed=elapsed,
                    error=exc,
                )

        # The loop should always return; this is a safeguard.
        raise RuntimeError("Evaluator execution loop exited unexpectedly.")

    @abstractmethod
    def _evaluate_impl(
        self,
        params: Mapping[str, Any],
        seed: int | None = None,
    ) -> Mapping[str, Any]:
        """Return the raw evaluator payload prior to normalization."""

    def __call__(
        self,
        params: Mapping[str, Any],
        seed: int | None = None,
        *,
        trial_timeout_sec: float | None = None,
        max_retries: int | None = None,
        graceful_nan_policy: GracefulNaNPolicy | str | None = None,
    ) -> EvaluatorResult:
        return self.evaluate(
            params,
            seed,
            trial_timeout_sec=trial_timeout_sec,
            max_retries=max_retries,
            graceful_nan_policy=graceful_nan_policy,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve_execution_config(
        self,
        *,
        trial_timeout_sec: float | None,
        max_retries: int | None,
        graceful_nan_policy: GracefulNaNPolicy | str | None,
    ) -> _ExecutionConfig:
        timeout = (
            trial_timeout_sec
            if trial_timeout_sec is not None
            else getattr(self, "trial_timeout_sec", None)
        )
        resolved_max_retries = (
            max_retries if max_retries is not None else getattr(self, "max_retries", 0)
        )
        if resolved_max_retries is None:
            resolved_max_retries = 0
        resolved_max_retries = max(0, int(resolved_max_retries))

        policy_candidate = (
            graceful_nan_policy
            if graceful_nan_policy is not None
            else getattr(self, "graceful_nan_policy", self.DEFAULT_NAN_POLICY)
        )
        policy = self._coerce_nan_policy(policy_candidate)

        if timeout is not None and timeout <= 0:
            timeout = 0.0

        return _ExecutionConfig(timeout=timeout, max_retries=resolved_max_retries, nan_policy=policy)

    def _coerce_nan_policy(
        self, candidate: GracefulNaNPolicy | str | None
    ) -> GracefulNaNPolicy:
        if isinstance(candidate, GracefulNaNPolicy):
            return candidate
        if isinstance(candidate, str):
            try:
                return GracefulNaNPolicy(candidate)
            except ValueError as exc:  # pragma: no cover - defensive branch
                raise ValueError(f"Unknown graceful_nan_policy: {candidate!r}") from exc
        return self.DEFAULT_NAN_POLICY

    def _execute_trial(
        self,
        *,
        params: Mapping[str, Any],
        seed: int | None,
        timeout: float | None,
    ) -> Mapping[str, Any]:
        def _invoke() -> Mapping[str, Any]:
            with self._evaluation_context(seed):
                return self._evaluate_impl(params, seed)

        if timeout is None:
            return _invoke()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_invoke)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError as exc:
                future.cancel()
                raise TimeoutError("Evaluator execution exceeded timeout") from exc

    @contextlib.contextmanager
    def _evaluation_context(self, seed: int | None):
        with contextlib.ExitStack() as stack:
            stack.enter_context(self._temporary_isolation())
            if seed is not None:
                stack.enter_context(self._seed_random_generators(seed))
            yield

    @contextlib.contextmanager
    def _temporary_isolation(self):
        with tempfile.TemporaryDirectory(prefix="astraia_eval_") as tmpdir:
            with contextlib.ExitStack() as stack:
                for var in ("TMPDIR", "TEMP", "TMP"):
                    stack.enter_context(self._env_swap(var, tmpdir))
                stack.enter_context(self._tempfile_override(tmpdir))
                yield tmpdir

    @contextlib.contextmanager
    def _tempfile_override(self, tmpdir: str):
        original = tempfile.tempdir
        tempfile.tempdir = tmpdir
        try:
            yield
        finally:
            tempfile.tempdir = original

    @contextlib.contextmanager
    def _env_swap(self, name: str, value: str):
        original = os.environ.get(name)
        os.environ[name] = value
        try:
            yield
        finally:
            if original is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = original

    @contextlib.contextmanager
    def _seed_random_generators(self, seed: int):
        random_state = random.getstate()
        random.seed(seed)

        numpy_state = torch_state = None
        numpy_module = self._safe_import("numpy")
        if numpy_module is not None:
            try:
                numpy_state = numpy_module.random.get_state()
                numpy_module.random.seed(seed)
            except AttributeError:  # pragma: no cover - numpy API differences
                numpy_state = None

        torch_module = self._safe_import("torch")
        if torch_module is not None:
            try:
                torch_state = torch_module.random.get_rng_state()
                torch_module.manual_seed(seed)
            except AttributeError:  # pragma: no cover - torch absent/random API
                torch_state = None

        try:
            yield
        finally:
            random.setstate(random_state)
            if numpy_module is not None and numpy_state is not None:
                try:
                    numpy_module.random.set_state(numpy_state)
                except AttributeError:  # pragma: no cover
                    pass
            if torch_module is not None and torch_state is not None:
                try:
                    torch_module.random.set_rng_state(torch_state)
                except AttributeError:  # pragma: no cover
                    pass

    def _safe_import(self, module_name: str):
        try:
            return __import__(module_name)
        except Exception:  # noqa: BLE001 - import may fail for optional deps
            return None

    def _finalize_result(
        self,
        payload: Mapping[str, Any],
        *,
        params: Mapping[str, Any],
        elapsed: float,
        nan_policy: GracefulNaNPolicy,
    ) -> EvaluatorResult:
        """Validate and normalize the raw evaluator payload."""

        normalized: MutableMapping[str, MetricValue]
        normalized = dict(payload)

        missing = [key for key in self.REQUIRED_METRICS if key not in normalized]
        if missing:
            raise ValueError(
                "Evaluator payload is missing required metrics: " + ", ".join(missing)
            )

        invalid_metrics: list[str] = []
        for metric in self.REQUIRED_METRICS:
            try:
                normalized[metric] = float(normalized[metric])  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise TypeError(f"Metric {metric!r} must be convertible to float") from exc

            value = float(normalized[metric])
            if math.isnan(value) or math.isinf(value):
                invalid_metrics.append(metric)

        if invalid_metrics:
            return self._handle_invalid_metrics(
                normalized,
                params=params,
                elapsed=elapsed,
                invalid_metrics=invalid_metrics,
                nan_policy=nan_policy,
            )

        status = normalized.get("status")
        if status is None:
            normalized["status"] = "ok"
        elif isinstance(status, str):
            if status not in self.VALID_STATUSES:
                raise ValueError(f"Unsupported evaluator status: {status!r}")
        else:
            raise TypeError("Evaluator status must be a string when provided.")

        for flag in ("timed_out", "terminated_early"):
            if flag not in normalized:
                normalized[flag] = False
            else:
                normalized[flag] = bool(normalized[flag])

        if "elapsed_seconds" in normalized and normalized["elapsed_seconds"] is not None:
            normalized["elapsed_seconds"] = float(normalized["elapsed_seconds"])
        else:
            normalized["elapsed_seconds"] = float(elapsed)

        if "reason" in normalized and normalized["reason"] is not None:
            normalized["reason"] = str(normalized["reason"])

        try:
            validated = EvaluatorOutput.model_validate(normalized)
        except ValidationError as exc:
            raise ValueError("Evaluator payload failed validation") from exc

        return validated.model_dump()

    def _handle_invalid_metrics(
        self,
        normalized: MutableMapping[str, MetricValue],
        *,
        params: Mapping[str, Any],
        elapsed: float,
        invalid_metrics: Iterable[str],
        nan_policy: GracefulNaNPolicy,
    ) -> EvaluatorResult:
        invalid_list = list(invalid_metrics)
        reason_suffix = ",".join(sorted(invalid_list))
        if nan_policy is GracefulNaNPolicy.ERROR:
            raise ValueError(
                "Evaluator payload produced non-finite metrics: " + ", ".join(invalid_list)
            )

        if nan_policy is GracefulNaNPolicy.COERCE_TO_INF:
            for metric in invalid_list:
                normalized[metric] = float("inf")
            original_status = normalized.get("status")
            if isinstance(original_status, str) and original_status in self.VALID_STATUSES and original_status != "ok":
                normalized["status"] = original_status
            else:
                normalized["status"] = "error"
            existing_reason = normalized.get("reason")
            normalized["reason"] = (
                str(existing_reason)
                if existing_reason is not None
                else f"invalid_metric:{reason_suffix}"
            )
            normalized["timed_out"] = bool(normalized.get("timed_out", False))
            normalized["terminated_early"] = bool(normalized.get("terminated_early", False))
            normalized["elapsed_seconds"] = float(elapsed)
            try:
                validated = EvaluatorOutput.model_validate(normalized)
            except ValidationError as exc:
                raise ValueError("Evaluator payload failed validation after coercion") from exc
            return validated.model_dump()

        # MARK_FAILURE fallback
        existing_reason = normalized.get("reason")
        original_status = normalized.get("status")
        if isinstance(original_status, str) and original_status in self.VALID_STATUSES and original_status != "ok":
            failure_status = original_status
        else:
            failure_status = "error"
        return self._fallback_payload(
            params=params,
            elapsed=elapsed,
            status=failure_status,
            reason=(
                str(existing_reason)
                if existing_reason is not None
                else f"invalid_metric:{reason_suffix}"
            ),
            payload=normalized,
            timed_out=bool(normalized.get("timed_out", False)),
        )

    def _fallback_payload(
        self,
        *,
        params: Mapping[str, Any],
        elapsed: float,
        status: str,
        reason: str,
        payload: Mapping[str, Any] | None = None,
        timed_out: bool = False,
    ) -> EvaluatorResult:
        param_count = float(len(params))
        depth = self._safe_float_from_payload(payload, "depth", default=param_count)
        shots = self._safe_float_from_payload(payload, "shots", default=0.0)
        result: EvaluatorResult = {
            "kl": float("inf"),
            "depth": depth,
            "shots": shots,
            "params": self._safe_float_from_payload(payload, "params", default=param_count),
            "status": status,
            "reason": reason,
            "timed_out": bool(timed_out),
            "terminated_early": False,
            "elapsed_seconds": float(elapsed),
        }
        validated = EvaluatorOutput.model_validate(result)
        return validated.model_dump()

    def _safe_float_from_payload(
        self,
        payload: Mapping[str, Any] | None,
        key: str,
        *,
        default: float,
    ) -> float:
        if payload is None or key not in payload:
            return float(default)
        try:
            value = float(payload[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return float(default)
        if math.isnan(value) or math.isinf(value):
            return float(default)
        return value

    def _timeout_result(
        self,
        *,
        params: Mapping[str, Any],
        elapsed: float,
    ) -> EvaluatorResult:
        return self._fallback_payload(
            params=params,
            elapsed=elapsed,
            status="timeout",
            reason="trial_timeout",
            timed_out=True,
        )

    def _exception_result(
        self,
        *,
        params: Mapping[str, Any],
        elapsed: float,
        error: Exception,
    ) -> EvaluatorResult:
        reason = f"exception:{error.__class__.__name__}"
        return self._fallback_payload(
            params=params,
            elapsed=elapsed,
            status="error",
            reason=reason,
            payload=None,
        )

