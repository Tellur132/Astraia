from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Mapping

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config_service import (
    ConfigLoadError,
    build_config_summary,
    get_config_root,
    list_config_entries,
    read_config_text,
    validate_config,
)
from .env_status import env_status
from .run_service import (
    LlmComparisonLaunch,
    RunLaunch,
    RunOptions,
    active_job_info,
    dry_run_config,
    list_runs as service_list_runs,
    load_run_detail,
    request_cancel,
    start_llm_comparison,
    start_run,
)
from astraia.config import ValidationError
from astraia.tracking import RunMetadata


class ConfigItem(BaseModel):
    name: str
    path: str
    tags: list[str] = Field(default_factory=list)


class ConfigListResponse(BaseModel):
    root: str
    items: list[ConfigItem]


class ConfigDetail(BaseModel):
    name: str
    path: str
    yaml: str


class ConfigJson(BaseModel):
    path: str
    config: dict[str, Any]


class ConfigSummary(BaseModel):
    path: str
    name: str
    summary: dict[str, Any]


class ValidationProblem(BaseModel):
    path: list[str]
    message: str


class ValidationErrorResponse(BaseModel):
    errors: list[ValidationProblem]


class RunOptionsPayload(BaseModel):
    max_trials: int | None = Field(default=None, ge=1)
    sampler: str | None = None
    llm_enabled: bool | None = None
    seed: int | None = Field(default=None, ge=0)


class DryRunRequest(BaseModel):
    config_path: str
    run_id: str | None = None
    options: RunOptionsPayload | None = None
    ping_llm: bool = True


class DryRunResponse(BaseModel):
    run_id: str
    config_path: str
    config: dict[str, Any]


class RunStartRequest(BaseModel):
    config_path: str
    run_id: str | None = None
    options: RunOptionsPayload | None = None
    perform_dry_run: bool = True
    ping_llm: bool = True
    llm_comparison: bool = False


class RunHandle(BaseModel):
    run_id: str
    status: str
    run_dir: str
    meta_path: str


class RunComparisonInfo(BaseModel):
    comparison_id: str
    shared_seed: int | None = None
    record_path: str | None = None
    llm_enabled: RunHandle
    llm_disabled: RunHandle


class RunStartResponse(BaseModel):
    run_id: str
    status: str
    run_dir: str
    meta_path: str
    comparison: RunComparisonInfo | None = None


class RunListItem(BaseModel):
    run_id: str
    status: str | None = None
    created_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    report: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    status_payload: dict[str, Any] = Field(default_factory=dict)
    run_dir: str | None = None


class JobInfo(BaseModel):
    pid: int | None = None
    state: str | None = None
    cancel_requested: bool | None = None
    started_at: str | None = None


class RunDetailResponse(BaseModel):
    meta: dict[str, Any]
    config: dict[str, Any] | None = None
    job: JobInfo | None = None


class CancelResponse(BaseModel):
    ok: bool
    message: str


def create_app() -> FastAPI:
    app = FastAPI(title="Astraia GUI Backend", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/configs", response_model=ConfigListResponse)
    def get_configs() -> ConfigListResponse:
        root = _config_root_or_error()
        items = [ConfigItem(**entry) for entry in list_config_entries(root)]
        return ConfigListResponse(root=str(root), items=items)

    @app.get("/configs/{config_path:path}", response_model=ConfigDetail)
    def get_config_yaml(config_path: str) -> ConfigDetail:
        root = _config_root_or_error()
        try:
            resolved, text = read_config_text(config_path, root=root)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ConfigLoadError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        rel_path = _relative_to_root(resolved, root)
        name = Path(rel_path).with_suffix("").as_posix()
        return ConfigDetail(name=name, path=rel_path, yaml=text)

    @app.get(
        "/configs/{config_path:path}/as-json",
        response_model=ConfigJson,
        responses={status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ValidationErrorResponse}},
    )
    def get_config_json(config_path: str) -> ConfigJson:
        root = _config_root_or_error()
        try:
            resolved, model = validate_config(config_path, root=root)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ConfigLoadError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except ValidationError as exc:
            detail = ValidationErrorResponse(errors=_format_validation_errors(exc))
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=detail.model_dump(),
            ) from exc

        rel_path = _relative_to_root(resolved, root)
        return ConfigJson(path=rel_path, config=model.model_dump(mode="json"))

    @app.get(
        "/configs/{config_path:path}/summary",
        response_model=ConfigSummary,
        responses={status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ValidationErrorResponse}},
    )
    def get_config_summary(config_path: str) -> ConfigSummary:
        root = _config_root_or_error()
        try:
            resolved, model = validate_config(config_path, root=root)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ConfigLoadError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except ValidationError as exc:
            detail = ValidationErrorResponse(errors=_format_validation_errors(exc))
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=detail.model_dump(),
            ) from exc

        rel_path = _relative_to_root(resolved, root)
        name = Path(rel_path).with_suffix("").as_posix()
        summary = build_config_summary(model)
        return ConfigSummary(path=rel_path, name=name, summary=summary)

    @app.get("/env/status")
    def get_env_status() -> dict[str, Any]:
        return env_status()

    @app.post(
        "/runs/dry-run",
        response_model=DryRunResponse,
        responses={status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ValidationErrorResponse}},
    )
    def post_runs_dry_run(payload: DryRunRequest) -> DryRunResponse:
        try:
            options = _options_from_payload(payload.options)
            result = dry_run_config(
                payload.config_path,
                run_id=payload.run_id,
                options=options,
                ping_llm=payload.ping_llm,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ConfigLoadError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except ValidationError as exc:
            detail = ValidationErrorResponse(errors=_format_validation_errors(exc))
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=detail.model_dump(),
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return DryRunResponse(
            run_id=result.run_id,
            config_path=str(result.config_path),
            config=result.config,
        )

    @app.post(
        "/runs",
        response_model=RunStartResponse,
        responses={status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ValidationErrorResponse}},
    )
    def post_runs(payload: RunStartRequest) -> RunStartResponse:
        try:
            options = _options_from_payload(payload.options)
            if payload.llm_comparison:
                comparison_launch = start_llm_comparison(
                    payload.config_path,
                    run_id=payload.run_id,
                    options=options,
                    perform_dry_run=payload.perform_dry_run,
                    ping_llm=payload.ping_llm,
                )
                return RunStartResponse(
                    run_id=comparison_launch.llm_enabled.run_id,
                    status=comparison_launch.llm_enabled.status,
                    run_dir=str(comparison_launch.llm_enabled.run_dir),
                    meta_path=str(comparison_launch.llm_enabled.meta_path),
                    comparison=_serialize_comparison_launch(comparison_launch),
                )

            launch = start_run(
                payload.config_path,
                run_id=payload.run_id,
                options=options,
                perform_dry_run=payload.perform_dry_run,
                ping_llm=payload.ping_llm,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ConfigLoadError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except ValidationError as exc:
            detail = ValidationErrorResponse(errors=_format_validation_errors(exc))
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=detail.model_dump(),
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return RunStartResponse(
            run_id=launch.run_id,
            status=launch.status,
            run_dir=str(launch.run_dir),
            meta_path=str(launch.meta_path),
        )

    @app.get("/runs", response_model=list[RunListItem])
    def get_runs(status_filter: str | None = None) -> list[RunListItem]:
        runs = service_list_runs(status_filter)
        return [_serialize_run_metadata(entry) for entry in runs]

    @app.get("/runs/{run_id}", response_model=RunDetailResponse)
    def get_run_detail(run_id: str) -> RunDetailResponse:
        try:
            metadata, config = load_run_detail(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

        job = active_job_info(metadata.run_id)
        job_payload = JobInfo(**job) if job else None
        return RunDetailResponse(meta=metadata.raw, config=config, job=job_payload)

    @app.post("/runs/{run_id}/cancel", response_model=CancelResponse)
    def cancel_run(run_id: str) -> CancelResponse:
        try:
            active = request_cancel(run_id)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

        message = "Cancellation signal sent." if active else "Run was already finished."
        return CancelResponse(ok=active, message=message)

    return app


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _format_validation_errors(exc: ValidationError) -> List[ValidationProblem]:
    errors: list[ValidationProblem] = []
    for error in exc.errors(include_url=False):
        loc = [str(part) for part in error.get("loc", ())]
        errors.append(ValidationProblem(path=loc, message=error.get("msg", "")))
    return errors


def _config_root_or_error() -> Path:
    try:
        return get_config_root()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc


def _options_from_payload(payload: RunOptionsPayload | None) -> RunOptions | None:
    if payload is None:
        return None
    return RunOptions(
        max_trials=payload.max_trials,
        sampler=payload.sampler,
        llm_enabled=payload.llm_enabled,
        seed=payload.seed,
    )

def _cors_origins() -> list[str]:
    env_value = os.environ.get("ASTRAIA_GUI_CORS_ORIGINS")
    if env_value:
        return [item.strip() for item in env_value.split(",") if item.strip()]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def _serialize_run_metadata(metadata: RunMetadata) -> RunListItem:
    artifacts = {
        key: str(value) for key, value in metadata.artifacts.items()
    } if isinstance(metadata.artifacts, Mapping) else {}
    status_payload = (
        dict(metadata.status_payload) if isinstance(metadata.status_payload, Mapping) else {}
    )
    return RunListItem(
        run_id=metadata.run_id,
        status=metadata.status,
        created_at=metadata.created_at.isoformat() if metadata.created_at else None,
        metadata=dict(metadata.metadata),
        report=dict(metadata.report),
        artifacts=artifacts,
        status_payload=status_payload,
        run_dir=str(metadata.run_dir),
    )


def _serialize_run_launch(launch: RunLaunch) -> RunHandle:
    return RunHandle(
        run_id=launch.run_id,
        status=launch.status,
        run_dir=str(launch.run_dir),
        meta_path=str(launch.meta_path),
    )


def _serialize_comparison_launch(launch: LlmComparisonLaunch) -> RunComparisonInfo:
    return RunComparisonInfo(
        comparison_id=launch.comparison_id,
        shared_seed=launch.seed,
        record_path=str(launch.summary_path) if launch.summary_path else None,
        llm_enabled=_serialize_run_launch(launch.llm_enabled),
        llm_disabled=_serialize_run_launch(launch.llm_disabled),
    )
