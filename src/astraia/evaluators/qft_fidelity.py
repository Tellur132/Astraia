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


@dataclass(slots=True)
class QFTFidelityEvaluator:
    """Compute fidelity of a parameterized QFT approximation."""

    num_qubits: int = 3

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
        depth = float(approx.depth() or 0)
        gate_count = float(len(approx.data))

        elapsed = time.perf_counter() - start
        return {
            "metric_fidelity": fidelity,
            "metric_depth": depth,
            "metric_gate_count": gate_count,
            "phase_scale": phase_scale,
            "include_swaps": include_swaps,
            "status": "ok",
            "elapsed_seconds": elapsed,
        }


def create_qft_fidelity_evaluator(config: Mapping[str, Any]) -> QFTFidelityEvaluator:
    num_qubits = int(config.get("num_qubits", 3))
    return QFTFidelityEvaluator(num_qubits=num_qubits)


__all__ = ["QFTFidelityEvaluator", "create_qft_fidelity_evaluator"]
