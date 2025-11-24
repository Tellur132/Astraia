"""Approximate QFT evaluator based on Qiskit's statevector simulator."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector, state_fidelity

from .base import EvaluatorResult


@dataclass(frozen=True)
class CircuitStats:
    """Lightweight summary of a Qiskit circuit for metric calculations."""

    depth: float
    gate_count: float
    t_gate_count: float
    single_qubit_gates: int
    multi_qubit_gates: int
    measurement_ops: int
    reset_ops: int


@dataclass(slots=True)
class SimpleNoiseModel:
    """Estimate an error probability from coarse circuit statistics.

    The model intentionally stays simple to keep evaluation fast:

    - Each single- and two-qubit gate contributes an independent error chance.
    - Measurements and resets can be given their own error rates.
    - An optional per-depth decoherence term penalises long circuits.
    """

    single_qubit_error_rate: float = 0.0
    two_qubit_error_rate: float = 0.0
    measurement_error_rate: float = 0.0
    reset_error_rate: float = 0.0
    decoherence_per_depth: float = 0.0

    def estimate_error_probability(self, stats: CircuitStats) -> float:
        def _survival(probability: float, count: int) -> float:
            probability = max(0.0, min(1.0, probability))
            return (1.0 - probability) ** max(count, 0)

        survival_terms = [
            _survival(self.single_qubit_error_rate, stats.single_qubit_gates),
            _survival(self.two_qubit_error_rate, stats.multi_qubit_gates),
            _survival(self.measurement_error_rate, stats.measurement_ops),
            _survival(self.reset_error_rate, stats.reset_ops),
            _survival(self.decoherence_per_depth, int(stats.depth)),
        ]

        survival = 1.0
        for term in survival_terms:
            survival *= term
        return max(0.0, min(1.0, 1.0 - survival))


def _prepare_input_state(num_qubits: int, value: int) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        if (value >> qubit) & 1:
            circuit.x(qubit)
    return circuit


def _approximate_qft(num_qubits: int, phase_scale: float, include_swaps: bool) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    for target in range(num_qubits):
        circuit.h(target)
        for control_offset in range(1, num_qubits - target):
            angle = phase_scale * math.pi / (2**control_offset)
            circuit.cp(angle, target + control_offset, target)
    if include_swaps:
        for i in range(num_qubits // 2):
            circuit.swap(i, num_qubits - i - 1)
    return circuit


def _circuit_stats(circuit: QuantumCircuit) -> CircuitStats:
    single_qubit = 0
    multi_qubit = 0
    t_gates = 0
    measurements = 0
    resets = 0

    for instruction, qargs, _ in circuit.data:
        name = getattr(instruction, "name", "").lower()
        qubit_count = len(qargs)

        if name == "t":
            t_gates += 1
        if name == "measure":
            measurements += 1
        if name == "reset":
            resets += 1

        if qubit_count <= 1:
            single_qubit += 1
        else:
            multi_qubit += 1

    return CircuitStats(
        depth=float(circuit.depth() or 0),
        gate_count=float(len(circuit.data)),
        t_gate_count=float(t_gates),
        single_qubit_gates=single_qubit,
        multi_qubit_gates=multi_qubit,
        measurement_ops=measurements,
        reset_ops=resets,
    )


@dataclass(slots=True)
class QFTFidelityEvaluator:
    """Compute fidelity of a parameterized QFT approximation."""

    num_qubits: int = 3
    noise_model: SimpleNoiseModel | None = None

    def __call__(self, params: Mapping[str, Any], seed: int | None = None) -> EvaluatorResult:  # noqa: ARG002 - seed reserved
        start = time.perf_counter()
        phase_scale = float(params.get("phase_scale", 1.0))
        include_swaps = bool(params.get("include_swaps", True))
        input_value = int(params.get("input_value", 1)) % (1 << self.num_qubits)

        prep = _prepare_input_state(self.num_qubits, input_value)
        approx = prep.compose(_approximate_qft(self.num_qubits, phase_scale, include_swaps))
        target = prep.compose(QFT(self.num_qubits, do_swaps=include_swaps)).decompose()

        approx_state = Statevector.from_instruction(approx)
        target_state = Statevector.from_instruction(target)

        fidelity = float(state_fidelity(approx_state, target_state))
        stats = _circuit_stats(approx)

        error_probability = None
        if self.noise_model is not None:
            error_probability = self.noise_model.estimate_error_probability(stats)

        elapsed = time.perf_counter() - start
        return {
            "metric_fidelity": fidelity,
            "metric_depth": stats.depth,
            "metric_gate_count": stats.gate_count,
            "metric_t_gate_count": stats.t_gate_count,
            "metric_error_probability": error_probability,
            "phase_scale": phase_scale,
            "include_swaps": include_swaps,
            "status": "ok",
            "elapsed_seconds": elapsed,
        }


def _build_noise_model(config: Mapping[str, Any] | None) -> SimpleNoiseModel | None:
    if not isinstance(config, Mapping):
        return None
    return SimpleNoiseModel(
        single_qubit_error_rate=float(config.get("single_qubit_error_rate", 0.0)),
        two_qubit_error_rate=float(config.get("two_qubit_error_rate", 0.0)),
        measurement_error_rate=float(config.get("measurement_error_rate", 0.0)),
        reset_error_rate=float(config.get("reset_error_rate", 0.0)),
        decoherence_per_depth=float(config.get("decoherence_per_depth", 0.0)),
    )


def create_qft_fidelity_evaluator(config: Mapping[str, Any]) -> QFTFidelityEvaluator:
    num_qubits = int(config.get("num_qubits", 3))
    noise_model = _build_noise_model(config.get("noise_model"))
    return QFTFidelityEvaluator(num_qubits=num_qubits, noise_model=noise_model)


def create_qft_synthesis_evaluator(config: Mapping[str, Any]) -> QFTFidelityEvaluator:
    """Factory with clearer naming for multi-objective synthesis tasks."""

    return create_qft_fidelity_evaluator(config)


__all__ = [
    "CircuitStats",
    "SimpleNoiseModel",
    "QFTFidelityEvaluator",
    "create_qft_fidelity_evaluator",
    "create_qft_synthesis_evaluator",
]
