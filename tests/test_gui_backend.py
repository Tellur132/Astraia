from __future__ import annotations

import pytest
import yaml
from fastapi.testclient import TestClient

from astraia_gui.backend import config_service
from astraia_gui.backend.app import create_app


def build_sample_config() -> dict:
    return {
        "metadata": {"name": "experiment", "description": "example"},
        "seed": 123,
        "search": {
            "library": "optuna",
            "sampler": "tpe",
            "n_trials": 4,
            "direction": "minimize",
            "metric": "kl",
        },
        "stopping": {
            "max_trials": 4,
            "max_time_minutes": 5,
            "no_improve_patience": 2,
        },
        "search_space": {
            "theta": {
                "type": "float",
                "low": -1.0,
                "high": 1.0,
            }
        },
        "evaluator": {"module": "astraia.evaluators.qgan_kl", "callable": "create_evaluator"},
        "report": {"metrics": ["kl"]},
    }


@pytest.fixture
def gui_client(tmp_path, monkeypatch) -> tuple[TestClient, str]:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_dir.joinpath("demo.yaml").write_text(
        yaml.safe_dump(build_sample_config()), encoding="utf-8"
    )

    monkeypatch.setenv("ASTRAIA_CONFIG_ROOT", str(config_dir))
    config_service.get_config_root.cache_clear()
    client = TestClient(create_app())
    yield client, str(config_dir)
    client.close()
    config_service.get_config_root.cache_clear()


def test_configs_listing_and_detail(gui_client) -> None:
    client, config_dir = gui_client

    response = client.get("/configs")
    assert response.status_code == 200
    payload = response.json()
    assert payload["root"] == config_dir
    assert payload["items"][0]["path"] == "demo.yaml"

    detail = client.get("/configs/demo.yaml")
    assert detail.status_code == 200
    assert "metadata" in detail.json()["yaml"]


def test_config_json_and_summary(gui_client) -> None:
    client, _ = gui_client

    as_json = client.get("/configs/demo.yaml/as-json")
    assert as_json.status_code == 200
    body = as_json.json()
    assert body["config"]["metadata"]["name"] == "experiment"

    summary = client.get("/configs/demo.yaml/summary")
    assert summary.status_code == 200
    summary_body = summary.json()
    assert summary_body["summary"]["search"]["metrics"] == ["kl"]
    assert summary_body["summary"]["stopping"]["max_trials"] == 4


def test_validation_errors_are_reported(gui_client, tmp_path) -> None:
    client, _ = gui_client

    bad = build_sample_config()
    bad["report"]["metrics"] = ["depth"]
    bad_path = tmp_path / "configs" / "invalid.yaml"
    bad_path.write_text(yaml.safe_dump(bad), encoding="utf-8")

    response = client.get("/configs/invalid.yaml/as-json")
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["errors"]
    assert any("report.metrics" in "/".join(item["path"]) for item in detail["errors"])


def test_env_status_marks_missing_keys(gui_client, monkeypatch) -> None:
    client, _ = gui_client
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "demo-secret")

    response = client.get("/env/status")
    assert response.status_code == 200
    payload = response.json()
    assert "OPENAI_API_KEY" in payload["missing_keys"]
    openai = payload["providers"]["openai"]
    assert not openai["ok"]

    gemini_entry = payload["providers"]["gemini"]["present"][0]
    assert gemini_entry["name"] == "GEMINI_API_KEY"
    assert gemini_entry["masked"].startswith("de")

