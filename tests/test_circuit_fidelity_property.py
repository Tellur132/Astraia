"""Property-based tests for circuit fidelity evaluator."""
from __future__ import annotations

from hypothesis import given, settings, strategies as st
from qiskit import qasm3
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Operator

from astraia.evaluators.circuit_fidelity import CircuitFidelityEvaluator


@settings(max_examples=10, deadline=2000)
@given(
    num_qubits=st.integers(min_value=1, max_value=3),
    depth=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=0, max_value=2**16 - 1),
)
def test_random_circuit_round_trip(num_qubits: int, depth: int, seed: int) -> None:
    circuit = random_circuit(num_qubits, depth, measure=False, seed=seed)
    unitary = Operator(circuit).data
    evaluator = CircuitFidelityEvaluator(target_unitary=unitary)

    qasm_code = qasm3.dumps(circuit)
    result = evaluator({"circuit_code": qasm_code})

    assert result["status"] == "ok"
    assert result["metric_valid"] == 1.0
    assert result["metric_fidelity"] >= 0.999
    assert result["metric_depth"] >= 0
    assert result["metric_gate_count"] >= 0
