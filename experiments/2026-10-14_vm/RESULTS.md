# 2026-10-14 — why the partial trace follows the mean (v ∥ m), and what that says about the skew law

The user asked to derive v ∥ m (measured in `../2026-10-13_prop3`: cos 0.92–0.99 at layers ≥ 8, n = 128/256) and to keep
working on the science. Notation: layer l, pre-activation z = Wᵀh₋ (W i.i.d. N(0, q/n), q = 2), post h = ReLU(z),
m = E h, μ = E z, σ² = Var z, α = μ/σ; v = the partial trace of the POST-activation third cumulant, vₐ = Σ_b κ₃(hₐ, h_b, h_b).

## 1. Theory (written before the runs)

**Identity (exact, any random vector).** κ(X, Y, Y) = Cov(X, Y²) − 2E[Y] Cov(X, Y), so
vₐ = Σ_b κ₃(hₐ, h_b, h_b) = Cov(hₐ, ‖h − m‖²): the partial trace is each neuron's covariance with the squared deviation
norm of the whole layer.

**Proposition 5 (the skew law is the alignment of v with m; exact in expectation over the next weights).** Let the next
layer's column w be i.i.d. N(0, q/n), independent of h. The next pre-activation κ₃ is t′ = Σ w_a w_b w_c κ₃(h)_abc and
its mean μ′ = wᵀm. By Isserlis, E_w[t′μ′] = 3(q/n)² vᵀm and E_w[μ′²] = (q/n)‖m‖², E t′ = E μ′ = 0, so the regression slope
of the next layer's κ₃ on its mean over the neurons is

  c(l+1) = (3q/n) · ⟨v_l, m_l⟩ / ‖m_l‖²,   i.e.   c(l+1)·n = 6 ⟨v, m⟩/‖m‖²,

and (with E t′² = (q/n)³(6‖κ₃‖² + 9‖v‖²), Prop. 3) R²(l+1) = cos²(v_l, m_l) · q_v/(1 + q_v). The skew law at layer l+1 is
exactly the statement that v is aligned with m at layer l; its slope is v's projection coefficient on m. Proposition 4
is the case l = 0 (v₀ = (κ₃(h) + s³B₂)·1 + a zero-mean part, m₀ ∝ 1).

**The input radius is an exact scale mode.** For x ~ N(0, Iₙ), x = ρ·x_s with ρ = ‖x‖/√n independent of the direction
x_s (uniform on the sphere of radius √n), and a bias-free ReLU network is positively homogeneous: z(x) = ρ·z(x_s) at
every layer. For a scale mixture z = s·ẑ with s independent of ẑ:
κ₃(z) = E[s³] κ₃(ẑ) + (E s³ − E s² E s) Sym(Ĉ ⊗ μ̂) + κ₃(s) μ̂⊗μ̂⊗μ̂. With E ρ² = 1: E ρ³ − E ρ = 1/n + O(n⁻²), κ₃(ρ) =
O(n⁻²). Hence, at EVERY layer, Gaussian-input κ₃ = sphere-input κ₃ + (3/n) σ̂ₐ² μ̂ₐ on the diagonal, and

  c_gauss(l)·n − c_sphere(l)·n = 3 σ̄²_w(l) + O(1/n),   σ̄²_w = Σₐ σₐ² μₐ(μₐ − μ̄) / Σₐ (μₐ − μ̄)²,

At layer 1, σ² = q·Var ReLU(N(0, q)) = 2(1 − 1/π) = 1.363 → 3σ² = 6(1 − 1/π) = 4.09 — exactly the ρ² term of
Proposition 4. So **Proposition 4's 13.91 splits into 9.82 from the projected per-neuron skews and 4.09 from the input's
radius**, and on the sphere c(1)·n = 9.82.

**Why v might be ∥ m beyond the radius (the scale response).** By Euler's theorem for ReLU, the response of E[ReLU(z)]
to a common rescaling z → (1 + η)z is exactly m = μΦ(α) + σφ(α); to a pure mean shift along μ it is μΦ(α); to a pure
change of spread it is σφ(α). A Gaussian expansion of vₐ = Cov(hₐ, ‖h − m‖²) to first order in C and κ₃ gives
vₐ = 2Φ(αₐ)(C₀u)ₐ + (φ(αₐ)/σₐ) Σ_b C²ₐ_b w_b + Φ(αₐ) Σ_b Tₐ_bb w_b + (φ(αₐ)/σₐ) Σ_b Dₐ_b u_b + κ₃(hₐ), with
u_b = m_b(1 − Φ(α_b)), w_b = Φ(α_b) − m_b φ(α_b)/σ_b: a combination of the two functions μΦ(α) (where C₀u and T·w point
along μ) and σφ(α). v ∥ m holds when the combination is the scale response (equal weights), and approximately whenever
the two functions are nearly collinear over the neurons, which they become as the spread of α grows with depth.

## 2. Predictions (pre-registered)

- P1 (Prop. 5): c_meas(l+1)·n = 6⟨v_l, m_l⟩/‖m_l‖² within the finite-n spread of the next layer's columns (a few %).
- P2 (radius): on the sphere, c(1)·n = 9.82; at every layer c_gauss·n − c_sphere·n = 3σ̄²_w(l) within MC error.
- P3: cos(v_l, m_l) grows with depth; most of it survives on the sphere at depth (the alignment is not only the radius).
- P4 (open, no direction registered): v_l ≈ λ₁ μΦ(α) + λ₂ σφ(α) with R² ≥ 0.8 at depth; whether λ₁ ≈ λ₂.

## 3. Measurement plan
`vm_mc.py <net> <gauss|sphere> <N> <seed>`: public 1024 × 16 networks, two passes over the same random stream (pass 1:
exact sample moments; pass 2: centred by the exact sample means, so vₐ = E[yₐ‖y‖²] needs no Gram matrix), per layer:
μ, σ², κ₃ (pre), m (post), v (post), the pre-activation partial trace, and C·m (is the mean an eigenvector of the
post covariance?).

## 4. Results, public 1024 × 16 networks 0 and 1 (2·10⁶ samples each; `res/analysis_1024.txt`, `res/analysis_*.json`)

Sanity: the MC post-means match the 10⁹ truth to 0.02–0.1% rms. Selected layers (network 0; network 1 alike, below):

| layer | c·n measured | c·n from v (P1) | sphere c·n | gauss − sphere | 3σ̄²_w (P2) | cos(v, m) gauss / sphere | εn gauss / sphere | Rayleigh m̂ᵀCm̂ / mean var | cos(Cm̂, m) | cos(v, Cm̂) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | — | — | — | — | — | 0.921 / 0.857 | 2.59 / 2.09 | 1.2 | 0.863 | 0.991 |
| 1 | 13.80 | 13.71 | 9.68 | 4.11 | 4.10 | 0.949 / 0.923 | 3.94 / 3.43 | 3.7 | 0.910 | 0.942 |
| 2 | 15.29 | 14.87 | 12.20 | 3.09 | 3.08 | 0.964 / 0.948 | 4.67 / 4.16 | 7.1 | 0.934 | 0.939 |
| 4 | 12.68 | 12.82 | 10.72 | 1.96 | 1.94 | 0.971 / 0.962 | 5.66 / 5.16 | 15.3 | 0.954 | 0.936 |
| 8 | 8.69 | 8.71 | 7.72 | 0.97 | 0.96 | 0.979 / 0.974 | 7.24 / 6.73 | 37.0 | 0.974 | 0.968 |
| 12 | 7.54 | 7.54 | 6.84 | 0.69 | 0.68 | 0.985 / 0.982 | 8.59 / 8.08 | 77.2 | 0.981 | 0.981 |
| 15 | 6.74 | 6.95 | 6.15 | 0.59 | 0.58 | 0.989 / 0.987 | 8.44 / 7.94 | 101.3 | 0.988 | 0.989 |

(ε = Var(β)/‖m‖², β = (h − m)·m̂: the relative variance of the representation's amplitude along the mean.)

- **P1 holds** [M]: c(l+1)·n measured against 6⟨v_l, m_l⟩/‖m_l‖²: max deviation 3.0% / 2.9% (gauss, networks 0 / 1), median
  0.6% / 1.6%; on the sphere max 3.6% / 3.8%. The skew law IS the alignment of the post-activation partial trace with
  the mean, quantitatively, at every layer.
- **P2 holds, essentially exactly** [M]: gauss − sphere slope = 3σ̄²_w at every layer to within 0.016 (network 0) and
  0.024 (network 1). On the sphere c(1)·n = 9.68 / 9.24 against the derived 9.82 (the Gaussian values, 13.80 / 13.37, sit
  equally below 13.91: the per-network spread, SD 0.62 over twenty networks). Also exact: εn on the sphere is lower by
  0.50 at every layer = the radius's Var(ρ)·n = 1/2. With the infinite-width ReLU kernel map
  f(ρ) = (√(1−ρ²) + (π − arccos ρ)ρ)/π, the mean-direction share ‖m‖²/E‖h‖² = f^(l+1)(0) to 1–3% (0.3183 exact at layer 0),
  so the radius's part of the slope is ≈ 6(1 − f^(l)(0)) — 4.09 at layer 1, 0.46 at layer 15 (measured 3σ̄²_w 0.59: the
  μ²-weighting).
- **P3 holds** [M]: the alignment survives without the radius (sphere 0.857 → 0.987); it is made inside the network.
- **The linear step transports v exactly** [M]: cos(Wᵀv_l, v^pre_(l+1)) = 0.999–1.000 at every layer (WWᵀ ≈ qI).
- **The mechanism, measured link by link.** (a) The mean is a near-eigenvector of the post-activation covariance
  (cos(Cm̂, m) 0.86 → 0.99) whose Rayleigh quotient grows from 1.2 to 101 times the mean per-neuron variance: an amplitude
  mode along the mean takes over with depth. (b) v is the response to that amplitude: v ≈ κ·Cm̂ with κ = E[βQ]/E[β²]
  (cos(v, Cm̂) 0.93–0.99, rising with depth). (c) The response of every neuron to a common rescaling of its input is
  exactly its mean (Euler's theorem for ReLU), so a pure amplitude mode would give Cm̂ = εm exactly; the measured
  cos(Cm̂, m) says how pure it is. (a)–(c) give cos(v, m) 0.92 → 0.99.
- **How pure is the scale mixture** [M]: the coupling of the amplitude to the orthogonal fluctuation energy,
  Cov(β, ‖y⊥‖²), is 62–78% of the pure-scale-mixture value 2ε‖m‖ tr C⊥ (rising with depth), and the skew of the
  amplitude, E β³, carries 0.2% → 14% of ⟨v, m̂⟩. In the two-function description, v ≈ λ₁ μΦ(α) + λ₂ σφ(α) with R² 0.73
  (layer 1) → 0.97 (layer 15) and λ₂/λ₁ = 1.3–1.9 (a pure rescaling would give 1): the amplitude fluctuation widens the
  neurons' spread more than a rescaling does. At depth the two functions are nearly collinear (cos(μΦ(α), m) 0.83 →
  0.998, sd(α) 0.7 → 3.1), so the departure from a pure rescaling costs little alignment there.

**Network 1** (same table, `res/analysis_1024.txt`): P1 max 2.9% / median 1.6%; P2 within 0.024 at every layer; sphere
c(1)·n 9.24; cos(v, m) 0.917 → 0.986 (sphere 0.848 → 0.983); Rayleigh 1.2 → 102; εn 2.49 → 7.81.

**Layer 0 in closed form** [D]: m ∝ 1 there, so β = Σₐ(hₐ − mₐ)/√n, Var β = s²(Var R + φ(0)²/2) (the second term from
Σ_b ρ²ₐ_b ≈ 1) and ‖m‖²/n = s²φ(0)², giving εn = [Var R + φ(0)²/2]/φ(0)² = π − ½ = 2.64 for Gaussian
input and π − 1 = 2.14 on the sphere (the radius's ½ removed); measured 2.59 / 2.49 and 2.09 / 1.99. With depth εn
grows by increments that shrink like the neurons' own variance (the diagonal term n Σ m²Var(h)/‖m‖⁴ falls from 2.1 to 0.2;
the increments are 0.2–1.4 of it) and saturates near 8 (8.4 / 7.8 at layer 15).

## 5. Synthetic shapes (`res/analysis_synth.txt`; three 256 × 16, three 256 × 32 (+ one on the sphere), one 512 × 16)

- P1: median deviation 2.4–4.6% per network, worst single layers 6–30% (deep layers of 256 × 32): the finite-n sampling
  of the next layer's 256 columns, ×2 the 1024 spread as expected. The prediction follows each network's own slope where
  the networks disagree most: at layer 31 of 256 × 32 the three networks measure 16.8 / 9.7 / 7.1 and Prop. 5 gives
  17.4 / 10.9 / 5.5 — the CV of 77% across networks found in `../2026-10-14_skew` is the networks' different ⟨v, m⟩.
- Alignment at every shape: cos(v, m) 0.94–0.96 at layer 1 → 0.97–0.996 at depth; cos(Cm̂, m) up to 0.997; Rayleigh
  quotient up to 111 (256 × 16), 123 (512 × 16), 159–178 (layer 31 of 256 × 32); εn saturates at 8–10.
- Sphere (256 × 32, network 0): c(1)·n 9.0 (Gaussian 13.2), alignment 0.898 → 0.986.

## 6. Verdict

1. **Proposition 5 is confirmed on the truth** at n = 1024 (two networks, both inputs, every layer, median 0.6–1.6%), and
   at n = 256–512 within the expected finite-n spread. The skew law is not a separate empirical regularity: the slope at
   layer l+1 is (3q/n)·⟨v_l, m_l⟩/‖m_l‖² and its R² is cos²(v_l, m_l)·q_v/(1+q_v). It also gives the 1/3 of D ≈ t/3 [D]: by
   the same Isserlis split, the v-term of the next (2,1) slice is (q/n)(Wᵀv)_b against 3(q/n)(Wᵀv)_b in the next diagonal,
   identical in every row a.
2. **Proposition 4 splits into named parts, confirmed by intervention**: 13.91 = 9.82 (the per-neuron skews projected
   through w³) + 4.09 = 6(1 − 1/π) (the input's radius, an exact scale mode by positive homogeneity). Removing the radius
   (sphere input) leaves 9.68 / 9.24; the gauss − sphere difference equals 3σ̄²_w at every layer to ≤ 0.024.
3. **v ∥ m, mechanism**: v = Cov(h, ‖h − m‖²) [exact]; ‖h − m‖² is driven by an amplitude mode along the mean, which takes
   over the post-activation covariance with depth (the mean becomes a near-eigenvector, Rayleigh quotient 1 → 100+) [M];
   a neuron's response to a common rescaling is its mean (Euler) [D], and the measured response is close to it
   (cos(Cm̂, m) 0.86 → 0.99) [M]; v is that response to 0.93–0.99 [M]. The radius is an exact instance (εn + ½); the rest of
   the mode is made by the network (sphere input keeps it). The mode is 62–78% of a pure scale mixture at 1024 × 16 (55–77% on the sphere, up to ≈ 100% at width 256): amplitude
   fluctuations widen the spread more than a rescaling (λ₂/λ₁ 1.3–1.9), which costs little alignment at depth because
   μΦ(α) and m converge as the spread of α grows (sd(α) 0.7 → 3.1).
4. Not derived: the growth and saturation of εn with depth (measured; the increments track the diagonal variance term
   only to a factor 0.2–1.4), and the 62–78% purity.

## 7. Files
`vm_mc.py` (MC, two passes), `vm_analyze.py` (all quantities above; needs the torch env only for synthetic networks);
`res/vm_*.npz` (per-layer vectors, gitignored), `res/analysis_1024.{txt,json}`, `res/analysis_synth.{txt,json}`,
`res/run_mc*.txt` (timings). Cost: ≈ 7 min per 1024 × 16 run at 2·10⁶ samples, ≈ 1–3 min per synthetic run; 12 runs
≈ 44 min in all. Figure: `writeup/short/fig_vm.png` (`writeup/short/make_figures.py`).

## 8. Fact-check (2026-10-14, independent sub-agent, read-only)
All derivations re-done and confirmed (Prop. 5 incl. the R² clause, by MC too; D′ corollary for the v-part; the identity; the
scale-mixture formula with E r³ − E r = E r/n exactly; B₂ = φ(0)·Var R so 3q²B₂/φ(0) = 6(1 − 1/π) identically; εn(0) = π − ½ /
π − 1; Euler; kernel map). Fixed: 12 runs (not 16); λ₂/λ₁ up to 1.9; the kernel-map share is 1–3% at 1024 but up to 9% at
256; the purity 62–78% holds for Gaussian input at 1024 (sphere 55–77%, 256-wide up to ≈ 1.0); wording "exactly in expectation
over the next layer's weights"; q_v and σ̄²_w defined; the radius renamed r (ρ stays the correlation).

## 9. Proposition 3 and Prop. 5's R² clause on the TRUE κ₃ tensor (`prop3_truth.py`, `p3t_summary.py`, `res/p3t_*.json`)
Until now Proposition 3 was checked on the chain's own dense tensor (`../2026-10-13_prop3`), an approximation of the truth.
Here G = κ₃ of the post-activations is estimated by Monte Carlo (n³ entries; two independent seeds of 2·10⁶ samples, each
centred by its exact sample means; every energy a cross product of the seeds), at layers 4, 8, 11, 14 of synthetic
networks of depth 16; the share is measured with the network's own next weights, as in prop3.

| n | (layer, net) pairs | share measured | share predicted | measured / predicted | f_AD | q_v | cos(v, m) | R² next, predicted | R² next, measured | slope: median |meas/pred − 1| |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 20 (5 nets) | 0.383 | 0.411 | **0.94 (sd 0.10)** | 0.66 | 2.97 | 0.888–0.993 | 0.685 | 0.704 (corr 0.78) | 14% |
| 128 | 12 (3 nets) | 0.381 | 0.377 | **1.01 (sd 0.13)** | 0.76 | 4.41 | 0.928–0.989 | 0.763 | 0.762 | 4.3% |
| 256 | 4 (1 net) | 0.337 | 0.346 | **0.97 (sd 0.04)** | 0.82 | 5.89 | 0.972–0.982 | 0.816 | 0.828 (corr 0.97; per layer 0.830/0.847, 0.819/0.836, 0.802/0.798, 0.815/0.832) | 3.8% |

On the chain's tensor prop3 had 0.94 ± 0.15 (n = 64), 1.00 ± 0.12 (n = 128) and 0.99 ± 0.06 (n = 256): the same. **Proposition 3 holds on the
truth, so the share the slices cannot carry is a property of the network family, not of our chain's arithmetic.** The R²
clause of Proposition 5 predicts the noise-corrected R² of the next layer's skew law on average to 0.02 (n = 64) and 0.001
(n = 128). Runtime: ≈ 40 s per network at n = 64, ≈ 5 min at n = 128, 17 min at n = 256 (two seeds of 2·10⁶ each). The n = 256 network ran twice: a first launch I
wrongly took for dead (`res/p3t_run256.txt`, 20:30–20:52, overlapping the activation runs) and the queued one
(`res/run_act_and_256.txt`); the numbers are identical, only the timing differs.

## 10. Other activations — which parts are general, which need positive homogeneity (pre-registered 2026-10-14 20:40)
Proposition 5 uses only the identity and Isserlis, so it should hold for ANY activation. The radius result (gauss − sphere
slope = 3σ̄²_w) and the Euler response need positive homogeneity, which leaky ReLU has and GELU / SiLU do not.
Predictions, 256 × 16 synthetic networks (weights of `coh.weights`, rescaled to q = 2/(1 + a²) for leaky ReLU with a = 0.2,
q = 2 for GELU and SiLU), Gaussian and sphere input:
- A1: P1 (Prop. 5) holds for all three activations to the same finite-n spread as ReLU at n = 256 (median ≲ 5%).
- A2: for leaky ReLU, gauss − sphere = 3σ̄²_w at every layer (as for ReLU); for GELU and SiLU it does NOT (the radius is not a
  scale mode of the network), i.e. deviations far above the ReLU's 0.02.
- A3 (direction only): cos(Cm̂, m), the Euler link, is lower for GELU/SiLU than for leaky ReLU at the same depth.

### Results (`act_summary.py` → `res/act_summary.json`; 256 × 16, three networks per activation, 2·10⁶ samples, both inputs)

| activation | Prop. 5: median |meas/pred − 1| (gauss / sphere) | radius: |(g − s) − 3σ̄²_w| / |g − s|, median (max) | cos(v, m) layer 1 / 8 / 15 | cos(Cm̂, m) layer 1 / 8 / 15 | cos(v, Cm̂) layer 1 / 8 / 15 | raw R²(κ₃ ~ μ) layer 4 / 8 / 15 |
|---|---|---|---|---|---|---|
| ReLU (q = 2) | 3.3% / — | — (1024: ≤ 0.024 absolute) | 0.950 / 0.983 / 0.986 | 0.897 / 0.976 / 0.986 | 0.922 / 0.980 / 0.992 | 0.81 / 0.85 / 0.83 |
| leaky ReLU 0.2 | 4.9% / 5.3% | **2.7% (4.9%)** | 0.927 / 0.966 / 0.980 | 0.836 / 0.945 / 0.970 | 0.903 / 0.964 / 0.985 | 0.72 / 0.80 / 0.80 |
| GELU | 4.0% / 4.8% | **76% (98%)** | 0.954 / 0.968 / 0.948 | 0.915 / 0.976 / 0.966 | 0.935 / 0.997 / 0.997 | 0.86 / 0.88 / 0.73 |
| SiLU | 4.5% / 4.9% | **93% (99.5%)** | 0.951 / 0.876 / **0.753** | 0.904 / 0.916 / **0.886** | 0.939 / 0.990 / 0.937 | 0.86 / 0.74 / **0.47** |

- **A1 holds**: Proposition 5 is activation-agnostic, as derived (same finite-n spread for all four).
- **A2 holds**: the input radius is an exact scale mode only for the positively homogeneous activations (leaky ReLU within
  2.7%, the remainder being the O(1/n) term below); for GELU and SiLU the Gaussian–sphere difference has nothing to do with
  3σ̄²_w (76–93% off).
- **A3, as stated, fails**: the Euler link cos(Cm̂, m) is NOT lower for GELU (0.915 / 0.976 / 0.966, above leaky ReLU at
  layers 1–8), and for SiLU it is lower only at depth (0.886 vs 0.970 at layer 15). What does separate the families is the
  alignment itself and with it the skew law: for the positively homogeneous activations cos(v, m) rises with depth to
  0.98–0.99 and the law's R² stays 0.80–0.83; for SiLU cos(v, m) falls to 0.75 and R² to 0.47 at layer 15 (GELU, nearly
  homogeneous for large |z|, loses a little: 0.95, 0.73). In SiLU it is the last link that breaks: v is still the response
  to the amplitude (cos(v, Cm̂) 0.94–0.99), but that response is not the mean. **The skew law at depth is a property of
  positively homogeneous activations; Proposition 5 is not.**
- Refinement of P2 (derived above to leading order): exactly, c_gauss·n − c_sphere·n = 3σ̄²_w + c_sphere + O(1/n); the extra
  c_sphere (≈ 0.01 at n = 1024, ≈ 0.05 at n = 256) is the residual seen in both the ReLU (0.016 / 0.024) and leaky-ReLU
  (+0.05–0.08) comparisons.
