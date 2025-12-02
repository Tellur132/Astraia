"""Simple QAOA evaluator using Qiskit's statevector simulator."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity

from .base import EvaluatorResult
from .exact_solvers import Edge, build_maxcut_operator, solve_cost_hamiltonian_ground_state
from .noise_simulation import NISQNoiseConfig, simulate_noisy_density_matrix


def _extract_angles(params: Mapping[str, Any], n_layers: int) -> Iterable[Tuple[float, float]]:
    for layer in range(n_layers):
        gamma = float(params.get(f"gamma_{layer}", params.get("gamma", 0.0)))
        beta = float(params.get(f"beta_{layer}", params.get("beta", 0.0)))
        yield gamma, beta


def _success_probability(state: Statevector | DensityMatrix, bitstrings: Sequence[str]) -> float:
    """Probability of measuring any optimal bitstring."""
    if isinstance(state, Statevector):
        return float(
            sum(abs(state.data[int(bits or "0", 2)]) ** 2 for bits in bitstrings)
        )
    if isinstance(state, DensityMatrix):
        return float(
            sum(np.real(state.data[int(bits or "0", 2), int(bits or "0", 2)]) for bits in bitstrings)
        )
    raise TypeError(f"Unsupported state type for success probability: {type(state)!r}")


def _bitstrings_to_statevector(bitstrings: Sequence[str], num_qubits: int) -> Statevector:
    """Uniform superposition over provided bitstrings."""
    dimension = 1 << num_qubits
    amplitudes = np.zeros(dimension, dtype=complex)
    weight = 1 / math.sqrt(max(len(bitstrings), 1))
    for value in bitstrings:
        index = int(value, 2) if value else 0
        amplitudes[index] = weight
    return Statevector(amplitudes)


@dataclass(slots=True)
class QAOAEvaluator:
    """Evaluate a small MaxCut instance with a QAOA ansatz."""

    num_qubits: int = 2
    edges: Tuple[Edge, ...] = ((0, 1),)
    noise_simulation: NISQNoiseConfig | None = None

    def __call__(self, params: Mapping[str, Any], seed: int | None = None) -> EvaluatorResult:  # noqa: ARG002 - seed reserved
        start = time.perf_counter()
        max_layers = 6
        n_layers = max(1, min(int(params.get("n_layers", 1)), max_layers))

        circuit = QuantumCircuit(self.num_qubits)
        circuit.h(range(self.num_qubits))
        for gamma, beta in _extract_angles(params, n_layers):
            for i, j in self.edges:
                circuit.rzz(2 * gamma, i, j)
            for qubit in range(self.num_qubits):
                circuit.rx(2 * beta, qubit)

        state = Statevector.from_instruction(circuit)

        cost_operator = build_maxcut_operator(self.num_qubits, self.edges)
        exact_energy, optimal_bitstrings = solve_cost_hamiltonian_ground_state(
            cost_operator, num_qubits=self.num_qubits
        )
        target = _bitstrings_to_statevector(optimal_bitstrings, self.num_qubits)
        # Lower energy (more negative) corresponds to larger cuts with this H_C.
        energy = float(np.real(state.expectation_value(cost_operator)))
        energy_gap = energy - exact_energy
        success_prob = _success_probability(state, optimal_bitstrings)

        fidelity = float(state_fidelity(state, target))

        depth = float(circuit.depth() or 0)
        gate_count = float(len(circuit.data))

        noise_metrics: dict[str, Any] = {}
        if self.noise_simulation is not None and self.noise_simulation.enabled:
            try:
                noisy_state = simulate_noisy_density_matrix(
                    circuit, self.noise_simulation, seed=seed
                )
                noisy_energy = float(np.real(noisy_state.expectation_value(cost_operator)))
                noisy_success = _success_probability(noisy_state, optimal_bitstrings)
                noisy_fidelity = float(state_fidelity(noisy_state, target))
                noise_metrics = {
                    "metric_energy_noisy": noisy_energy,
                    "metric_fidelity_noisy": noisy_fidelity,
                    "metric_energy_delta": noisy_energy - energy,
                    "metric_fidelity_delta": fidelity - noisy_fidelity,
                    "metric_success_prob_opt_noisy": noisy_success,
                    "metric_success_prob_opt_delta": noisy_success - success_prob,
                    "noise_model_label": self.noise_simulation.label,
                    "noise_status": "ok",
                }
            except Exception as exc:  # noqa: BLE001 - surfaced to caller
                noise_metrics = {
                    "noise_status": "error",
                    "noise_model_label": self.noise_simulation.label,
                    "noise_error": str(exc),
                }

        elapsed = time.perf_counter() - start
        return {
            "metric_fidelity": fidelity,
            "metric_energy": energy,
            "metric_energy_exact": exact_energy,
            "metric_energy_gap": energy_gap,
            "metric_success_prob_opt": success_prob,
            "metric_depth": depth,
            "metric_gate_count": gate_count,
            "n_layers": float(n_layers),
            "status": "ok",
            "elapsed_seconds": elapsed,
            **noise_metrics,
        }


def create_qaoa_evaluator(config: Mapping[str, Any]) -> QAOAEvaluator:
    num_qubits = int(config.get("num_qubits", 2))
    edges = tuple(tuple(edge) for edge in config.get("edges", ((0, 1),)))
    noise_cfg = NISQNoiseConfig.from_mapping(config.get("noise_simulation"))
    return QAOAEvaluator(
        num_qubits=num_qubits,
        edges=edges,  # type: ignore[arg-type]
        noise_simulation=noise_cfg,
    )


__all__ = ["QAOAEvaluator", "create_qaoa_evaluator"]
