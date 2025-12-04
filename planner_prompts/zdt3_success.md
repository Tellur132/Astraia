Example 1 - fast HV warm-up (trial < 6)
{
  "objectives": ["f1", "f2"],
  "emphasis": "spread",
  "parameter_focus": {
    "x1": {"mode": "lhs", "range_hint": [0.05, 0.95]},
    "x2": {"mode": "lhs", "range_hint": [0.05, 0.95]},
    "x3": {"mode": "lhs", "range_hint": [0.05, 0.95]},
    "x4": {"mode": "lhs", "range_hint": [0.05, 0.95]},
    "x5": {"mode": "lhs", "range_hint": [0.05, 0.95]}
  },
  "batch_size_hint": 3,
  "notes": "Latin-hypercube style spread across [0,1]^5; ensure at least one point near each extreme of x1 to cover discontinuous fronts."
}

---

Example 2 — refinement after HV > 0.5*best
{
  "objectives": ["f1", "f2"],
  "emphasis": "balance",
  "parameter_focus": {
    "x1": {"mode": "exploit", "range_hint": [0.1, 0.35]},
    "x2": {"mode": "exploit", "range_hint": [0.1, 0.35]},
    "x3": {"mode": "exploit", "range_hint": [0.1, 0.35]},
    "x4": {"mode": "explore", "range_hint": [0.0, 0.7]},
    "x5": {"mode": "explore", "range_hint": [0.0, 0.7]}
  },
  "batch_size_hint": 3,
  "notes": "Nudge x1-x3 toward mid-low values to land on non-dominated bands; keep a wide spread on x4/x5 to avoid collapsing onto a single segment."
}
