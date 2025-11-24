from __future__ import annotations

from astraia.evaluators.noise_simulation import NISQNoiseConfig


def test_from_mapping_builds_config() -> None:
    cfg = NISQNoiseConfig.from_mapping(
        {
            "label": "device_mock",
            "enabled": False,
            "single_qubit_depolarizing": 0.5,
            "two_qubit_depolarizing": 2.0,  # will be clamped
            "readout_error": -1.0,  # will be clamped
            "method": "density_matrix",
            "seed_simulator": 123,
        }
    )
    assert cfg is not None
    assert cfg.label == "device_mock"
    assert cfg.enabled is False
    clamped = cfg.clamped()
    assert clamped.two_qubit_depolarizing <= 1.0
    assert clamped.readout_error >= 0.0


def test_from_mapping_rejects_non_mapping() -> None:
    assert NISQNoiseConfig.from_mapping(None) is None
