"""Tests for the QAOA evaluator energy calculations."""
from __future__ import annotations

import unittest

from astraia.evaluators.exact_solvers import (
    build_maxcut_operator,
    solve_cost_hamiltonian_ground_state,
    solve_maxcut_exact,
)
from astraia.evaluators.qaoa import QAOAEvaluator


class QAOAEvaluatorEnergyTests(unittest.TestCase):
    def test_metric_energy_exact_uses_same_cost_hamiltonian(self) -> None:
        evaluator = QAOAEvaluator(num_qubits=2, edges=((0, 1),))
        params = {"gamma_0": 0.0, "beta_0": 0.0}

        metrics = evaluator(params, seed=123)

        self.assertIn("metric_energy_exact", metrics)
        self.assertIn("metric_energy_gap", metrics)
        self.assertIn("metric_success_prob_opt", metrics)

        cost_operator = build_maxcut_operator(2, ((0, 1),))
        exact_energy, bitstrings = solve_cost_hamiltonian_ground_state(
            cost_operator, num_qubits=2
        )

        # Ground energy for a single edge is -1.0 with the H_C used for metric_energy.
        self.assertAlmostEqual(exact_energy, -1.0)
        self.assertCountEqual(bitstrings, ["01", "10"])
        self.assertAlmostEqual(metrics["metric_energy_exact"], exact_energy)

        # Non-optimal angles should sit above the exact ground energy (less negative).
        self.assertGreater(metrics["metric_energy"], metrics["metric_energy_exact"])
        self.assertAlmostEqual(metrics["metric_energy_gap"], 0.5)
        self.assertAlmostEqual(metrics["metric_success_prob_opt"], 0.5)

    def test_maxcut_exact_returns_all_degenerate_bitstrings(self) -> None:
        edges = ((0, 1), (1, 2), (2, 3), (3, 0))
        energy, bitstrings = solve_maxcut_exact(4, edges)

        self.assertAlmostEqual(energy, -4.0)
        self.assertCountEqual(bitstrings, ["0101", "1010"])


if __name__ == "__main__":
    unittest.main()
