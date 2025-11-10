"""Fallback implementation for key Pydantic interfaces used in the project."""

from __future__ import annotations

import sys
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)


T = TypeVar("T", bound="BaseModel")
_UNSET = object()


class ValidationError(Exception):
    """Exception raised when validation fails."""

    def __init__(self, errors: List[Dict[str, Any]]):
        super().__init__("Validation failed")
        self._errors = errors

    def errors(self, *, include_url: bool | None = None) -> List[Dict[str, Any]]:
        return self._errors


def ConfigDict(**kwargs: Any) -> Dict[str, Any]:
    return dict(kwargs)


def Field(*args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - shim
    return dict(kwargs)


def constr(**kwargs: Any) -> Type[str]:  # pragma: no cover - shim
    return str


def conint(**kwargs: Any) -> Type[int]:  # pragma: no cover - shim
    return int


def confloat(**kwargs: Any) -> Type[float]:  # pragma: no cover - shim
    return float


@dataclass
class ModelValidator:
    func: Callable[[Any], Any]
    mode: str


def model_validator(*, mode: str = "after") -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
    def decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
        setattr(func, "__pydantic_validator__", ModelValidator(func=func, mode=mode))
        return func

    return decorator


def field_validator(*fields: str, mode: str = "after") -> Callable[[Callable[[Any, Any], Any]], Callable[[Any, Any], Any]]:
    def decorator(func: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
        setattr(func, "__pydantic_field_validator__", {"fields": fields, "mode": mode})
        return func

    return decorator


@dataclass
class _FieldInfo:
    name: str
    annotation: Any
    default: Any
    required: bool


class _ModelMeta(type):
    def __new__(mcls, name: str, bases: Tuple[type, ...], namespace: Dict[str, Any]):
        module = sys.modules.get(namespace.get("__module__", "__main__"))
        cls = super().__new__(mcls, name, bases, namespace)

        fields: Dict[str, _FieldInfo] = {}
        for base in bases:
            fields.update(getattr(base, "__fields__", {}))

        hints = get_type_hints(
            cls,
            globalns=getattr(module, "__dict__", {}),
            localns=dict(vars(cls)),
        )

        for field_name, annotation in hints.items():
            if field_name == "model_config":
                continue
            default = getattr(cls, field_name, _UNSET)
            required = default is _UNSET
            default_value = None if required else default
            if hasattr(cls, field_name):
                try:
                    delattr(cls, field_name)
                except AttributeError:
                    pass
            fields[field_name] = _FieldInfo(
                name=field_name,
                annotation=annotation,
                default=default_value,
                required=required,
            )

        cls.__fields__ = fields

        validators: List[ModelValidator] = []
        for base in bases:
            validators.extend(getattr(base, "__model_validators__", []))

        for value in vars(cls).values():
            marker = getattr(value, "__pydantic_validator__", None)
            if marker:
                validators.append(marker)

        cls.__model_validators__ = validators

        field_validators: Dict[str, List[Tuple[str, Callable[[Any, Any], Any]]]] = {}
        for base in bases:
            if hasattr(base, "__field_validators__"):
                for field, items in base.__field_validators__.items():
                    field_validators.setdefault(field, []).extend(items)

        for value in vars(cls).values():
            marker = getattr(value, "__pydantic_field_validator__", None)
            if marker:
                for field in marker["fields"]:
                    field_validators.setdefault(field, []).append((marker["mode"], value))

        cls.__field_validators__ = field_validators
        return cls


class BaseModel(metaclass=_ModelMeta):
    model_config: Dict[str, Any] = ConfigDict()

    def __init__(self, **data: Any) -> None:
        errors: List[Dict[str, Any]] = []
        values: Dict[str, Any] = {}
        provided = dict(data)

        for field_name, field in self.__fields__.items():
            if field_name in provided:
                raw_value = provided.pop(field_name)
            elif not field.required:
                raw_value = field.default
            else:
                errors.append({"loc": (field_name,), "msg": "Field required"})
                continue

            try:
                value = self._coerce_value(field.annotation, raw_value)
            except ValidationError as exc:
                for err in exc.errors():
                    errors.append({"loc": (field_name,) + err.get("loc", ()), "msg": err.get("msg", "Invalid value")})
                continue
            except Exception as exc:  # pragma: no cover - defensive
                errors.append({"loc": (field_name,), "msg": str(exc)})
                continue

            for mode, validator in self.__field_validators__.get(field_name, []):
                if mode != "after":
                    continue
                try:
                    value = validator(self, value)
                except Exception as exc:
                    errors.append({"loc": (field_name,), "msg": str(exc)})
                    break
            else:
                values[field_name] = value

        extra_behavior = self.__class__.model_config.get("extra", "ignore")
        self.__extra__: Dict[str, Any] = {}
        if provided:
            if extra_behavior == "forbid":
                for name in provided:
                    errors.append({"loc": (name,), "msg": "Extra fields not permitted"})
            elif extra_behavior == "allow":
                self.__extra__ = provided

        if errors:
            raise ValidationError(errors)

        self.__dict__.update(values)
        if self.__extra__:
            self.__dict__.update(self.__extra__)

        for validator in self.__model_validators__:
            if validator.mode != "after":
                continue
            try:
                result = validator.func(self)
                if isinstance(result, self.__class__):
                    self.__dict__.update(result.__dict__)
            except ValidationError:
                raise
            except Exception as exc:
                raise ValidationError([{"loc": ("__root__",), "msg": str(exc)}]) from exc

    @classmethod
    def _coerce_value(cls, annotation: Any, value: Any) -> Any:
        origin = get_origin(annotation)
        if origin is None:
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                if not isinstance(value, dict):
                    raise TypeError("Value must be a mapping")
                return annotation.model_validate(value)
            if annotation in {int, float, bool, str}:
                if annotation is bool:
                    if isinstance(value, bool):
                        return value
                    if isinstance(value, str):
                        return value.lower() in {"1", "true", "yes"}
                    return bool(value)
                try:
                    return annotation(value)
                except Exception as exc:
                    raise TypeError(str(exc))
            return value

        if origin in {list, List}:
            (item_type,) = get_args(annotation) or (Any,)
            if not isinstance(value, list):
                raise TypeError("Value must be a list")
            return [cls._coerce_value(item_type, item) for item in value]

        if origin in {dict, Dict, MappingABC}:
            key_type, value_type = get_args(annotation) or (Any, Any)
            if not isinstance(value, dict):
                raise TypeError("Value must be a dict")
            coerced: Dict[Any, Any] = {}
            for key, item in value.items():
                coerced_key = key if key_type is Any else key_type(key)
                coerced[coerced_key] = cls._coerce_value(value_type, item)
            return coerced

        if origin is Iterable:
            return list(value)

        if origin is tuple:
            return tuple(value)

        if origin is Union:
            args = get_args(annotation)
            if type(None) in args and value is None:
                return None
            for arg in args:
                if arg is type(None):
                    continue
                try:
                    return cls._coerce_value(arg, value)
                except Exception:
                    continue
            raise TypeError("Value does not match any type in the Union")

        return value

    @classmethod
    def model_validate(cls: Type[T], data: Any) -> T:
        if not isinstance(data, dict):
            raise ValidationError([{"loc": ("__root__",), "msg": "Data must be a mapping"}])
        return cls(**data)

    def model_dump(self, *, mode: str = "python") -> Dict[str, Any]:  # pragma: no cover - simple
        result: Dict[str, Any] = {}
        for field_name in self.__fields__:
            value = getattr(self, field_name)
            result[field_name] = self._dump_value(value)
        for key, value in getattr(self, "__extra__", {}).items():
            result[key] = self._dump_value(value)
        return result

    @classmethod
    def _dump_value(cls, value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, list):
            return [cls._dump_value(item) for item in value]
        if isinstance(value, dict):
            return {key: cls._dump_value(item) for key, item in value.items()}
        return value


__all__ = [
    "BaseModel",
    "ConfigDict",
    "Field",
    "ValidationError",
    "confloat",
    "conint",
    "constr",
    "field_validator",
    "model_validator",
]
