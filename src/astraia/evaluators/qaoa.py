"""Simple QAOA evaluator using Qiskit's statevector simulator."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity

from .base import EvaluatorResult

Edge = Tuple[int, int]


def _build_maxcut_operator(num_qubits: int, edges: Sequence[Edge]) -> SparsePauliOp:
    terms: list[SparsePauliOp] = []
    for i, j in edges:
        z_term = ["I"] * num_qubits
        z_term[i] = "Z"
        z_term[j] = "Z"
        pauli = "".join(reversed(z_term))
        terms.append(SparsePauliOp(pauli, coeffs=[0.5]))
        terms.append(SparsePauliOp("I" * num_qubits, coeffs=[-0.5]))
    if not terms:
        return SparsePauliOp("I" * num_qubits, coeffs=[0.0])
    operator = terms[0]
    for term in terms[1:]:
        operator += term
    return operator


def _maxcut_best_state(num_qubits: int, edges: Sequence[Edge]) -> Statevector:
    if num_qubits == 0:
        return Statevector(np.array([1.0]))

    best_score = -math.inf
    best_strings: list[int] = []
    for bitstring in range(1 << num_qubits):
        score = 0
        for i, j in edges:
            bit_i = (bitstring >> i) & 1
            bit_j = (bitstring >> j) & 1
            if bit_i != bit_j:
                score += 1
        if score > best_score:
            best_score = score
            best_strings = [bitstring]
        elif score == best_score:
            best_strings.append(bitstring)

    amplitudes = np.zeros(1 << num_qubits, dtype=complex)
    weight = 1 / math.sqrt(max(len(best_strings), 1))
    for bitstring in best_strings:
        amplitudes[bitstring] = weight
    return Statevector(amplitudes)


def _extract_angles(params: Mapping[str, Any], n_layers: int) -> Iterable[Tuple[float, float]]:
    for layer in range(n_layers):
        gamma = float(params.get(f"gamma_{layer}", params.get("gamma", 0.0)))
        beta = float(params.get(f"beta_{layer}", params.get("beta", 0.0)))
        yield gamma, beta


@dataclass(slots=True)
class QAOAEvaluator:
    """Evaluate a small MaxCut instance with a QAOA ansatz."""

    num_qubits: int = 2
    edges: Tuple[Edge, ...] = ((0, 1),)

    def __call__(self, params: Mapping[str, Any], seed: int | None = None) -> EvaluatorResult:  # noqa: ARG002 - seed reserved
        start = time.perf_counter()
        n_layers = max(1, min(3, int(params.get("n_layers", 1))))

        circuit = QuantumCircuit(self.num_qubits)
        circuit.h(range(self.num_qubits))
        for gamma, beta in _extract_angles(params, n_layers):
            for i, j in self.edges:
                circuit.rzz(2 * gamma, i, j)
            for qubit in range(self.num_qubits):
                circuit.rx(2 * beta, qubit)

        state = Statevector.from_instruction(circuit)

        cost_operator = _build_maxcut_operator(self.num_qubits, self.edges)
        energy = float(np.real(state.expectation_value(cost_operator)))

        target = _maxcut_best_state(self.num_qubits, self.edges)
        fidelity = float(state_fidelity(state, target))

        depth = float(circuit.depth() or 0)
        gate_count = float(len(circuit.data))

        elapsed = time.perf_counter() - start
        return {
            "metric_fidelity": fidelity,
            "metric_energy": energy,
            "metric_depth": depth,
            "metric_gate_count": gate_count,
            "n_layers": float(n_layers),
            "status": "ok",
            "elapsed_seconds": elapsed,
        }


def create_qaoa_evaluator(config: Mapping[str, Any]) -> QAOAEvaluator:
    num_qubits = int(config.get("num_qubits", 2))
    edges = tuple(tuple(edge) for edge in config.get("edges", ((0, 1),)))
    return QAOAEvaluator(num_qubits=num_qubits, edges=edges)  # type: ignore[arg-type]


__all__ = ["QAOAEvaluator", "create_qaoa_evaluator"]
