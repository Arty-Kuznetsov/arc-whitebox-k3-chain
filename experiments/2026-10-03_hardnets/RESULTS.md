# Which networks are hard for us? (2026-10-03) — nothing simple

The leaders' advantage grows on the networks where our error is largest (INTEL-peers §5.2). On our 30 baked local
networks (`hardnets.py`, research base B9 + marg2 for the features, production k3z raw from
`../2026-10-02_k3z/runs/k3z_n100.json`): Spearman correlations of our raw with per-network features at layers 8, 12, 15
— the fractions of dead (α < −2.5), kink (|α| < 1) and saturated (α > 2.5) neurons, the κ₄ level v, the rms skewness —
are all within ±0.26 (kink at layer 12 +0.26, dead at 12 −0.25; n = 30). Network hardness is not a simple
activation-regime or non-Gaussianity statistic; the 12% spread of our error across networks has no cheap predictor.
