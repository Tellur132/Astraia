Example 1 - shallow angles stabilize quickly (HV lifts by trial 6)
{
  "objectives": ["energy", "depth"],
  "emphasis": "energy",
  "parameter_focus": {
    "n_layers": {"mode": "narrow", "range_hint": [1, 2]},
    "gamma_0": {"mode": "exploit", "range_hint": [2.8, 3.3]},
    "beta_0": {"mode": "exploit", "range_hint": [2.8, 3.3]},
    "gamma_1": {"mode": "freeze", "range_hint": [0.0, 0.5]},
    "beta_1": {"mode": "freeze", "range_hint": [0.0, 0.5]}
  },
  "batch_size_hint": 4,
  "notes": "Keep p=1 unless energy stagnates; sweep small steps around gamma_0/beta_0~3.1 while enforcing diversity."
}

---

Example 2 - rescue depth when energy plateaus
{
  "objectives": ["energy", "depth"],
  "emphasis": "depth",
  "parameter_focus": {
    "n_layers": {"mode": "toggle", "choices": [1, 2]},
    "gamma_0": {"mode": "shrink", "range_hint": [1.8, 2.5]},
    "beta_0": {"mode": "shrink", "range_hint": [1.8, 2.5]},
    "gamma_1": {"mode": "explore", "range_hint": [0.0, 1.2]},
    "beta_1": {"mode": "explore", "range_hint": [0.0, 1.2]}
  },
  "batch_size_hint": 3,
  "notes": "If depth balloons without HV gain, pivot to p=1 with smaller angles; only open p=2 after HV growth resumes."
}
