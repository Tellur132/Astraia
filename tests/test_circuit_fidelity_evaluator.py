"""Tests for the circuit fidelity evaluator that parses LLM outputs."""
from __future__ import annotations

import math
import unittest

import numpy as np

from astraia.evaluators.circuit_fidelity import CircuitFidelityEvaluator, create_circuit_fidelity_evaluator


class CircuitFidelityEvaluatorTests(unittest.TestCase):
    def test_qasm_truth_table_fidelity(self) -> None:
        evaluator = CircuitFidelityEvaluator(truth_table={"0": "1", "1": "0"})
        code = """```qasm
OPENQASM 3;
qubit[1] q;
x q[0];
```"""
        result = evaluator({"circuit_code": code})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["metric_valid"], 1.0)
        self.assertAlmostEqual(result["metric_fidelity"], 1.0, places=6)
        self.assertGreaterEqual(result["metric_gate_count"], 1.0)

    def test_python_code_unitary_target(self) -> None:
        target_unitary = np.array([[0, 1], [1, 0]], dtype=complex)
        evaluator = create_circuit_fidelity_evaluator({"target_unitary": target_unitary})
        code = """from qiskit import QuantumCircuit
qc = QuantumCircuit(1)
qc.x(0)
"""
        result = evaluator({"circuit_code": code})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["metric_valid"], 1.0)
        self.assertAlmostEqual(result["metric_fidelity"], 1.0, places=6)

    def test_invalid_code_returns_penalty(self) -> None:
        evaluator = CircuitFidelityEvaluator(truth_table={"0": "0"})
        result = evaluator({"circuit_code": "not valid quantum code"})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["metric_valid"], 0.0)
        self.assertEqual(result["metric_fidelity"], 0.0)
        self.assertTrue(math.isinf(result["metric_depth"]))
        self.assertTrue(math.isinf(result["metric_gate_count"]))


if __name__ == "__main__":
    unittest.main()
