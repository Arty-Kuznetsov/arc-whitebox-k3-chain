# The last layer alone at 1024×16 (2026-10-03) — what its moments are worth, and whether a corrector can take it

Motivation: at small width the next order's last-layer statistics alone are worth ×0.57 (`../2026-10-03_korder/`
§5) and the last layer has no downstream, so a correction there cannot break the chain's compensation.

## 1. The oracle bound (`ll_oracle.py`; research base B9 + marg2; Monte-Carlo pre-activation moments of layer 15,
1e7 samples × 2 seeds; Richardson 2P − (A+B)/2 removes the oracle's noise)
| substituted at layer 15 only | MLP 0 | MLP 1 |
|---|---|---|
| var | ×0.899 | ×0.915 |
| t (κ₃ diagonal) | ×0.922 | ×0.963 |
| K4d (κ₄ diagonal) | ×0.951 | ×0.979 |
| t + K4d | ×0.870 | ×0.930 |
| **var + t + K4d** | **×0.761** | **×0.815** |

## 2. A last-layer corrector, in sample (`ll_proj.py`): the errors' projection on the neuron's 21 own smooth features
| removed at layer 15 only | MLP 0 | MLP 1 |
|---|---|---|
| t: exact / projection (R² 0.51 / 0.48) / half | ×0.923 / **×0.999** / ×0.994 | ×0.964 / **×1.010** / ×0.999 |
| K4d: exact / projection (R² 0.57 / 0.51) / half | ×0.952 / **×0.992** / ×0.993 | ×0.979 / **×1.004** / ×0.999 |

**Closed.** Half of each error is predictable and removing it buys nothing even in sample, at the one layer where no
consistency can be broken: the damaging part of the last layer's t and κ₄ errors is the rough, network-specific part —
§70's lesson (R² is not the payoff) holds without propagation. The variance channel (×0.90 / ×0.92) is the random
projection of ΔC₀ (§72) and has no predictable part at all.
