"""Utilities for simple NISQ-like noise simulation with Qiskit Aer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Statevector


@dataclass(slots=True)
class NISQNoiseConfig:
    """Configuration for a coarse NISQ noise model."""

    label: str = "nisq"
    enabled: bool = True
    single_qubit_depolarizing: float = 0.0
    two_qubit_depolarizing: float = 0.0
    readout_error: float = 0.0
    method: str = "density_matrix"
    seed_simulator: int | None = None

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "NISQNoiseConfig | None":
        if not isinstance(config, Mapping):
            return None
        return cls(
            label=str(config.get("label", "nisq")),
            enabled=bool(config.get("enabled", True)),
            single_qubit_depolarizing=float(config.get("single_qubit_depolarizing", 0.0)),
            two_qubit_depolarizing=float(config.get("two_qubit_depolarizing", 0.0)),
            readout_error=float(config.get("readout_error", 0.0)),
            method=str(config.get("method", "density_matrix")),
            seed_simulator=(
                int(config["seed_simulator"])
                if config.get("seed_simulator") is not None
                else None
            ),
        )

    def clamped(self) -> "NISQNoiseConfig":
        def _clamp(value: float) -> float:
            return max(0.0, min(1.0, value))

        return NISQNoiseConfig(
            label=self.label,
            enabled=self.enabled,
            single_qubit_depolarizing=_clamp(self.single_qubit_depolarizing),
            two_qubit_depolarizing=_clamp(self.two_qubit_depolarizing),
            readout_error=_clamp(self.readout_error),
            method=self.method,
            seed_simulator=self.seed_simulator,
        )


def _build_noise_model(config: NISQNoiseConfig):
    try:
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    except Exception as exc:  # noqa: BLE001 - surfacing missing dependency
        raise RuntimeError("qiskit-aer is required for noise simulation") from exc

    cfg = config.clamped()
    model = NoiseModel()

    if cfg.single_qubit_depolarizing > 0.0:
        single_error = depolarizing_error(cfg.single_qubit_depolarizing, 1)
        model.add_all_qubit_quantum_error(
            single_error, ["x", "sx", "h", "rz", "p", "id"]
        )

    if cfg.two_qubit_depolarizing > 0.0:
        two_error = depolarizing_error(cfg.two_qubit_depolarizing, 2)
        model.add_all_qubit_quantum_error(
            two_error, ["cx", "cz", "swap", "rzz"]
        )

    if cfg.readout_error > 0.0:
        p = cfg.readout_error
        readout = ReadoutError([[1 - p, p], [p, 1 - p]])
        model.add_all_qubit_readout_error(readout)

    return model


def simulate_noisy_density_matrix(
    circuit: QuantumCircuit, config: NISQNoiseConfig, *, seed: int | None = None
) -> DensityMatrix:
    """Run a circuit through Aer with a coarse NISQ noise model."""

    if not config.enabled:
        raise ValueError("Noise simulation disabled in configuration")

    try:
        from qiskit_aer import AerSimulator
    except Exception as exc:  # noqa: BLE001 - surfacing missing dependency
        raise RuntimeError("qiskit-aer is required for noise simulation") from exc

    simulator = AerSimulator(
        method=config.method or "density_matrix",
        noise_model=_build_noise_model(config),
    )

    compiled = transpile(circuit, simulator, optimization_level=0)
    sim_seed = seed if seed is not None else config.seed_simulator
    result = simulator.run(compiled, seed_simulator=sim_seed).result()
    data = result.data(0)

    density_like = (
        data.get("density_matrix")
        or data.get("final_density_matrix")
        or data.get("statevector")
        or data.get("final_statevector")
    )
    if density_like is None:
        raise RuntimeError("Simulator did not return a statevector or density matrix")

    try:
        return DensityMatrix(density_like)
    except Exception:  # noqa: BLE001 - fallback for statevector payloads
        return DensityMatrix(Statevector(density_like))


__all__ = ["NISQNoiseConfig", "simulate_noisy_density_matrix"]
