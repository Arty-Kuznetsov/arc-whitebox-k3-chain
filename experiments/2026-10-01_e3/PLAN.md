# E3 (receiver rows) — the port plan, written 2026-10-01 before the work

**What E3 is** (cloud #8, `chain.py` lines 628–647, switch `recv=(θ_out, arm)`): neurons whose α at the CURRENT layer
is below θ_out = −3.0 ("receivers") do not get their leg columns computed at all. Their κ₃ diagonal t_i is filled in
instead from (a) the newest source alone and (b) the exact repeated-index rule on the previous layer's (t, D) through
the gate d1 — arm "b". Measured by the cloud: raw ×1.0020, **projected util ×0.953** on the full stack (our k3q meter
puts it at ≈×0.96 → **4.05e-9 → ≈3.90e-9**).

## Why it is safe downstream (the reason the cut is only one layer deep)
θ_out = −3.0 < deadcut = −2.5, so every receiver is already **dead**:
- its D rows *and* columns are already zeroed by the matched cut (`deadcut` + `colcut`), so **the pairing never sees
  them** and needs no change;
- its leg columns are multiplied by `d1 * mask_cur` = 0 at the end of the layer anyway
  (`fnp.multiply(XT[...], d1[None, :], ...)`), and the next layer's transport gathers only the alive columns
  (`cut_prev`), so **nothing downstream reads them**.
The *only* thing that changes is **this layer's readout of t_i for those neurons**, which is what the fill-in replaces.
At the last layer there is no gating, so the fill-in has to be good there too — which is what the cloud's ×1.0020 on
the full chain already measures.

## The three pieces

1. **The column cut of the transport** (`src/estimator_k3r.py`, the block "(2) the transport"). Today:
   `rsh(GX, Wg, XN[:Rp], K_, c_)` with `Wg` (npp, n). Change to `Wgk = fnp.take(Wg, idck, axis=1)` → (npp, nk) and
   `rsh(GX, Wgk, XNk[:Rp], ...)`, saving **Rp × npp × (n − nk)** products. The newborn's hub leg
   (`fnp.take(w, Pn, axis=0, out=XN[R:R+wn])`) takes the same columns. The band members' materialisation
   (`rsh(ZS, PT, XM, ...)`) inherits the narrower PT automatically.
2. **The scatter back.** Use the pattern `alive_cut` already uses for rows: build `cmap` = position of each neuron in
   `idck`, with the receivers mapped to one appended ZERO column, then `XN = fnp.take(XNk_padded, cmap, axis=1)`.
   Price: one (Rp × n) `take` per layer ≈ 3.6M elements, i.e. ~0.013% of the meter but ≈1–2 ms of residual per layer
   (~20 ms over the run, against 153 ms of grader headroom). The saving is ≈0.10·npp ≈ 90× the take, so it pays.
   ⚠️ Keep the legs in **neuron order**. Storing them in α-descending order would make the alive set a prefix and
   remove the scatter entirely, but it permutes every downstream index (readout, gating, next transport) — not worth it.
3. **The fill-in of t for the receivers (arm b).** From `chain.py`:
   `t_r = t_newest + (W∘W∘W) @ (d1³ t_prev) + 3 · ((W∘W) @ Dt) ∘ W summed over the contracted axis`,
   with `Dt = (d1²)[:,None] · D_prev · d1[None,:]`. **Compute it only on the receiver rows**: a (n_recv × n) @ (n × n)
   product, i.e. ≈0.10·n³ per layer ≈ 0.2% of the budget — it must NOT be computed for all n, which would cost 3.9%
   and eat the whole saving. Requires carrying `t_prev`, `D_prev` and `d1_prev` one layer (chain.py's `recv_state`).

## Verification (the project's standing discipline)
- `recv=None` must be **bitwise** k3r with equal FLOPs at 1024×16 and 128×8.
- Against the cloud's torch `recv=(-3.0, "b")` on a 128×8 MLP: agreement to 2–7e-7 per layer, as every previous port
  (k3o 2–6e-7, k3p 2–7e-7, k3q 2–7e-7).
- Then the 10-MLP screen on top of k3r, then 100 MLPs if raw ≤ ×1.005 and util ≤ ×0.97.

## Priced against the alternative
E3 is ≈×0.96 (rank 32 → 31). It is the last cheap exact lever: after it the engineering track is finished at
≈3.9e-9 and everything else must come from accuracy (`notes/INTEL-peers-2026-10-01.md`). Do it, but after the
cross-MLP corrector measurement, which decides where the remaining two weeks go.
