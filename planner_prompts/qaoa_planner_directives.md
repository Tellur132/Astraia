# QAOA planner directives for the MaxCut ring

- Target graph: 4 qubits arranged in a ring with edges (0-1), (1-2), (2-3), (3-0).
- Ansatz: standard QAOA layer per p with RZZ(2*gamma_l) on all edges then RX(2*beta_l) on every qubit.
- Objectives: primary is lowering expected cut energy; secondary is reducing circuit depth / gate_count without losing too much fidelity.
- Layer budget: n_layers is allowed in [1,4]; keep recommendations shallow unless the history shows energy plateaus.
- Heuristics: symmetry between early/late layers and small-angle starts (|gamma|, |beta| < 1.6) often help; widen ranges only after initial convergence.
- Planner output should steer batches toward the most sensitive angles first (gamma_0/beta_0), then expand to deeper layers if improvement stalls.
