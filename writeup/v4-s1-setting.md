## 1. Setting, metric and harness

**Task.** Layer *l* computes y<sup>(l)</sup> = ReLU(W<sup>(l)</sup>ᵀ y<sup>(l−1)</sup>) with W<sup>(l)</sup> ∼ N(0, 2/n) i.i.d., width n = 1024,
depth L = 16, and input y<sup>(−1)</sup> ∼ N(0, I<sub>n</sub>). The estimator returns E[y<sup>(l)</sup>] for every layer. The score is the final-layer MSE against an
N = 10⁹ Monte-Carlo ground truth, times `max(0.1, C/B)`, with `C` the FLOPs billed by flopscope and `B = 2⁴¹`, averaged
over networks. All arithmetic goes through `flopscope.numpy`; the grader zeroes a network's prediction if its
non-flopscope ("residual") time exceeds 0.4 s, its wall time 120 s, or its memory 8 GB (Appendix A.2).

**Notation.** For a pre-activation `z` with mean μ and covariance C: σᵢ = √Cᵢᵢ, αᵢ = μᵢ/σᵢ, and under the Gaussian
closure d₁ = E[ReLU′(zᵢ)] = Φ(αᵢ), d₂ = E[ReLU″(zᵢ)] = φ(αᵢ)/σᵢ. C0 is C with its diagonal removed; t = κ₃ᵢᵢᵢ;
D = κ₃ᵢᵢⱼ (the (2,1) slice); K22 = κ₄ᵢᵢⱼⱼ. "Raw" means the final-layer MSE; "util" means C/B.

**The baseline everyone starts from** is covariance propagation ("k = 2": the linear step exact, a Gaussian closure at
every ReLU): raw 4.05e-6 at 2.35% of the budget on the 100 public networks, i.e. adjusted 4.05e-7. Plain sampling at the
same metric bottoms out near 1.2e-6. The method of §3 reaches raw 2.03e-8 at 18.3% of the budget (adjusted 3.74e-9) —
200× more accurate than covariance propagation.

**Harness and discipline.** Every number in this document is a logged measurement against the baked 10⁹ ground truth
of the public `mini` split (100 networks) unless it is labelled otherwise. Three instruments are used:
- the **production estimator** under the official runner (`whest run`), which gives exact FLOPs and the score;
- a **float64 research chain** in torch with every term as a switch, used for ablations and oracle substitutions
  (it reproduces ARC's reference implementation slice by slice to 1e-15, and the production estimator to ≤1e-6);
- **Monte-Carlo oracles** of the chain's intermediate state (per-neuron moments, the full post-activation covariance,
  the (2,1) slice) at 10⁷ samples with two independent seeds, so that an oracle's own noise is measured and
  subtracted rather than assumed away.

Every switch, when off, reproduces its predecessor bitwise (production) or to float rounding (research chain); every
new measurement harness is first gated on a number it must reproduce. Those gates caught, among others, a
double-applied ReLU gate, a wrong layer's weight matrix in a delegated analysis (a bogus 138% effect), unstandardised
regression features, and a float64 overflow at dead neurons — each before it could become a claim. We use **[M]** for
a measured claim, **[D]** for one derived algebraically or computed from measured numbers, and **[U]** for our
interpretation where an intervention has not isolated it.

**Two machines.** Most measurements ran on one laptop; long batch jobs (oracles, catalogue screens) ran in Claude Code
cloud sessions with the same pinned code. FLOP counts and MSEs agree between the two to the printed digit; wall and
residual times do not transfer, so every timing decision was taken on the laptop and confirmed from the grader's own
telemetry, which every submission page publishes.
