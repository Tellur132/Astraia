"""Exact solvers for small discrete cost Hamiltonians."""
from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

Edge = Tuple[int, int]


def build_maxcut_operator(
    num_qubits: int, edges: Sequence[Edge], weights: Sequence[float] | None = None
) -> SparsePauliOp:
    """Cost Hamiltonian H_C = Σ w (0.5 * Z_i Z_j - 0.5 * I) for (weighted) MaxCut."""
    if weights is not None and len(weights) != len(edges):
        raise ValueError("weights must match the number of edges")

    terms: list[SparsePauliOp] = []
    for idx, (i, j) in enumerate(edges):
        weight = 1.0 if weights is None else float(weights[idx])
        z_term = ["I"] * num_qubits
        z_term[i] = "Z"
        z_term[j] = "Z"
        pauli = "".join(reversed(z_term))
        terms.append(SparsePauliOp(pauli, coeffs=[0.5 * weight]))
        terms.append(SparsePauliOp("I" * num_qubits, coeffs=[-0.5 * weight]))

    if not terms:
        return SparsePauliOp("I" * num_qubits, coeffs=[0.0])

    operator = terms[0]
    for term in terms[1:]:
        operator += term
    return operator


def solve_cost_hamiltonian_ground_state(
    cost_operator: SparsePauliOp, *, num_qubits: int | None = None, atol: float = 1e-12
) -> tuple[float, list[str]]:
    """Return the minimum eigenvalue and all optimal bitstrings for a diagonal H_C.

    Assumes the Hamiltonian is diagonal in the computational basis (e.g., MaxCut).
    """
    nq = num_qubits if num_qubits is not None else int(cost_operator.num_qubits)
    if nq < 0:
        raise ValueError("num_qubits must be non-negative")
    if nq == 0:
        state = Statevector(np.array([1.0]))
        energy = float(np.real(state.expectation_value(cost_operator)))
        return energy, [""]

    dimension = 1 << nq
    best_energy = math.inf
    best_strings: list[int] = []

    for bitstring in range(dimension):
        amplitudes = np.zeros(dimension, dtype=complex)
        amplitudes[bitstring] = 1.0
        basis_state = Statevector(amplitudes)
        energy = float(np.real(basis_state.expectation_value(cost_operator)))

        if energy < best_energy - atol:
            best_energy = energy
            best_strings = [bitstring]
        elif math.isclose(energy, best_energy, rel_tol=atol, abs_tol=atol):
            best_strings.append(bitstring)

    bitstrings = [format(value, f"0{nq}b") for value in best_strings]
    return best_energy, bitstrings


def solve_maxcut_exact(
    num_qubits: int, edges: Sequence[Edge], weights: Sequence[float] | None = None
) -> tuple[float, list[str]]:
    """Convenience wrapper to solve MaxCut exactly using the shared H_C definition."""
    cost_operator = build_maxcut_operator(num_qubits, edges, weights)
    return solve_cost_hamiltonian_ground_state(cost_operator, num_qubits=num_qubits)


__all__ = [
    "Edge",
    "build_maxcut_operator",
    "solve_cost_hamiltonian_ground_state",
    "solve_maxcut_exact",
]
