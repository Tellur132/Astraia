"""Tests for the QAOA evaluator energy calculations."""
from __future__ import annotations

import unittest

from astraia.evaluators import exact_solvers
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

    def test_exact_solution_cache_reuses_results(self) -> None:
        exact_solvers._MAXCUT_CACHE.clear()
        edges = ((0, 1),)
        first_energy, first_bits = solve_maxcut_exact(2, edges)
        cache_size = len(exact_solvers._MAXCUT_CACHE)
        second_energy, second_bits = solve_maxcut_exact(2, edges)

        self.assertEqual(cache_size, len(exact_solvers._MAXCUT_CACHE))
        self.assertEqual(first_energy, second_energy)
        self.assertEqual(first_bits, second_bits)

    def test_exact_computation_skips_when_over_max_qubits(self) -> None:
        evaluator = QAOAEvaluator(
            num_qubits=3,
            edges=((0, 1), (1, 2), (2, 0)),
            exact_solution_max_qubits=1,
        )
        metrics = evaluator({"gamma_0": 0.0, "beta_0": 0.0})

        self.assertIn("metric_energy_exact", metrics)
        self.assertIsNone(metrics["metric_energy_exact"])
        self.assertIsNone(metrics["metric_energy_gap"])
        self.assertIsNone(metrics["metric_success_prob_opt"])
        self.assertIsNone(metrics["metric_fidelity"])

    def test_maxcut_exact_returns_all_degenerate_bitstrings(self) -> None:
        edges = ((0, 1), (1, 2), (2, 3), (3, 0))
        energy, bitstrings = solve_maxcut_exact(4, edges)

        self.assertAlmostEqual(energy, -4.0)
        self.assertCountEqual(bitstrings, ["0101", "1010"])

    def test_ring_exact_energy_and_success_probability_bounds(self) -> None:
        edges = ((0, 1), (1, 2), (2, 3), (3, 0))
        energy, bitstrings = solve_maxcut_exact(4, edges)
        self.assertAlmostEqual(energy, -4.0)
        self.assertCountEqual(bitstrings, ["0101", "1010"])

        evaluator = QAOAEvaluator(num_qubits=4, edges=edges)
        metrics = evaluator({"gamma_0": 0.0, "beta_0": 0.0})

        self.assertAlmostEqual(metrics["metric_energy_exact"], -4.0)
        success = metrics["metric_success_prob_opt"]
        self.assertIsNotNone(success)
        self.assertGreaterEqual(success, 0.0)
        self.assertLessEqual(success, 1.0)


if __name__ == "__main__":
    unittest.main()
