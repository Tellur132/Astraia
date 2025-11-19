"""Evaluator that scores LLM-proposed quantum circuits against a target."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from qiskit import QuantumCircuit, qasm2, qasm3
from qiskit.quantum_info import Operator, Statevector, process_fidelity

from .base import EvaluatorResult


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
    return stripped


def _parse_circuit_from_code(code: str) -> QuantumCircuit:
    errors: list[str] = []
    cleaned = _strip_code_fences(code)

    # Try OpenQASM 3
    try:
        return qasm3.loads(cleaned)
    except Exception as exc:  # noqa: BLE001 - aggregated failure
        errors.append(f"qasm3: {exc}")

    # Try OpenQASM 2
    try:
        return qasm2.loads(cleaned)
    except Exception as exc:  # noqa: BLE001 - aggregated failure
        errors.append(f"qasm2: {exc}")

    # Try executing as Python code that defines a QuantumCircuit
    namespace: dict[str, Any] = {"QuantumCircuit": QuantumCircuit}
    try:
        exec(cleaned, namespace, namespace)  # noqa: S102 - trusted sandbox for evaluator
    except Exception as exc:  # noqa: BLE001
        errors.append(f"python: {exc}")
    else:
        for value in namespace.values():
            if isinstance(value, QuantumCircuit):
                return value
        errors.append("python: no QuantumCircuit found in executed code")

    raise ValueError("Failed to parse circuit code; attempts: " + "; ".join(errors))


def _normalise_truth_table(table: Mapping[str, str]) -> tuple[dict[str, str], int]:
    normalised: dict[str, str] = {}
    num_qubits: int | None = None
    for input_bits, output_bits in table.items():
        key = str(input_bits).strip()
        value = str(output_bits).strip()
        if num_qubits is None:
            num_qubits = len(key)
        if len(key) != num_qubits or len(value) != num_qubits:
            raise ValueError("All truth table entries must use consistent bitstring lengths")
        normalised[key] = value
    if num_qubits is None:
        raise ValueError("Truth table must not be empty")
    return normalised, num_qubits


def _unitary_from_config(target_unitary: Any) -> tuple[np.ndarray, int]:
    matrix = np.asarray(target_unitary, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("target_unitary must be a square matrix")
    dimension = matrix.shape[0]
    num_qubits = int(math.log2(dimension))
    if 2**num_qubits != dimension:
        raise ValueError("target_unitary dimension must be a power of two")
    return matrix, num_qubits


@dataclass(slots=True)
class CircuitFidelityEvaluator:
    """Evaluate candidate circuits against a target unitary or truth table."""

    target_unitary: np.ndarray | None = None
    truth_table: Mapping[str, str] | None = None
    num_qubits: int | None = None

    def __post_init__(self) -> None:
        if self.target_unitary is None and self.truth_table is None:
            raise ValueError("Either target_unitary or truth_table must be provided")
        if self.target_unitary is not None:
            matrix, num_qubits = _unitary_from_config(self.target_unitary)
            self.target_unitary = matrix
            self.num_qubits = num_qubits
        if self.truth_table is not None:
            table, table_qubits = _normalise_truth_table(self.truth_table)
            self.truth_table = table
            self.num_qubits = self.num_qubits or table_qubits
            if self.num_qubits != table_qubits:
                raise ValueError("target_unitary and truth_table must use the same qubit count")

    def __call__(self, params: Mapping[str, Any], seed: int | None = None) -> EvaluatorResult:  # noqa: ARG002 - seed reserved
        start = time.perf_counter()
        code = str(params.get("circuit_code", "")).strip()
        if not code:
            return self._failure_payload("missing_circuit_code", start)

        try:
            circuit = _parse_circuit_from_code(code)
        except Exception as exc:  # noqa: BLE001 - handled by evaluator
            return self._failure_payload(str(exc), start)

        if self.num_qubits is not None and circuit.num_qubits != self.num_qubits:
            return self._failure_payload(
                f"qubit_count_mismatch: expected {self.num_qubits}, got {circuit.num_qubits}",
                start,
            )

        try:
            fidelity = self._compute_fidelity(circuit)
        except Exception as exc:  # noqa: BLE001 - handled by evaluator
            return self._failure_payload(str(exc), start)

        depth = float(circuit.depth() or 0)
        gate_count = float(len(circuit.data))
        elapsed = time.perf_counter() - start
        return {
            "metric_fidelity": float(fidelity),
            "metric_depth": depth,
            "metric_gate_count": gate_count,
            "metric_valid": 1.0,
            "status": "ok",
            "elapsed_seconds": elapsed,
        }

    def _compute_fidelity(self, circuit: QuantumCircuit) -> float:
        if self.target_unitary is not None:
            target_op = Operator(self.target_unitary)
            circuit_op = Operator(circuit)
            return float(process_fidelity(circuit_op, target_op))
        if self.truth_table is None:
            raise ValueError("No target specified for fidelity computation")

        scores = []
        for input_bits, output_bits in self.truth_table.items():
            input_state = Statevector.from_label(input_bits)
            final_state = input_state.evolve(circuit)
            target_index = int(output_bits, 2)
            probability = abs(final_state.data[target_index]) ** 2
            scores.append(probability)
        return float(sum(scores) / len(scores))

    def _failure_payload(self, reason: str, start_time: float) -> EvaluatorResult:
        elapsed = time.perf_counter() - start_time
        return {
            "metric_fidelity": 0.0,
            "metric_depth": float("inf"),
            "metric_gate_count": float("inf"),
            "metric_valid": 0.0,
            "status": "error",
            "reason": reason,
            "elapsed_seconds": elapsed,
        }


def create_circuit_fidelity_evaluator(config: Mapping[str, Any]) -> CircuitFidelityEvaluator:
    target_unitary = config.get("target_unitary")
    truth_table = config.get("truth_table")
    return CircuitFidelityEvaluator(target_unitary=target_unitary, truth_table=truth_table)


__all__ = ["CircuitFidelityEvaluator", "create_circuit_fidelity_evaluator"]
