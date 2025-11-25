from __future__ import annotations

from pathlib import Path
from typing import Any, List

from fastapi import FastAPI, HTTPException, status
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
from astraia.config import ValidationError


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


def create_app() -> FastAPI:
    app = FastAPI(title="Astraia GUI Backend", version="0.1.0")

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
