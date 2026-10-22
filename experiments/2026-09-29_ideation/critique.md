# Ideation workflow — completeness critic (verbatim; translated from Russian on 2026-10-13)

**Gaps in the completeness of the synthesis: 5 missed directions**

I changed nothing and ran nothing except one numpy computation of about a second. It ran over the 50 per-MLP error values that already sit in `<repo>/experiments/2026-09-24_k3-chain/leaders/per_mlp_final.py`.

**The assumption shared by all seven lenses.** The chain is treated as a causal filter running forward in depth, with the same accuracy everywhere. Every live source is read at every layer. Every cutoff threshold is the same for all source ages. The constraints (wall, grader resolution) are treated as constants. Gaps 1, 2, 4 and 5 follow from this assumption. Gap 3 is an unused tool. One more observation: all seven lenses looked for cost reductions. The synthesis itself admits in §0.1 that we also fall short of the leaders in accuracy, but no lens looked for the channel in which we carry excess error.

---

**1. Transport need not coincide with readout: multi-layer ("jump") products for the old tail**

- **Idea.** A source's legs are transported layer by layer only because they are read at every layer. But all W are known in advance, and the chain computes the d1 gates of the intermediate layers itself.
  - The old tier (age ≥ a0) is read exactly only at the checkpoint layers c, c+k, … The last layer is always a checkpoint.
  - Between them the old tier's legs are not materialised. The operator shared by all sources, J = M_{c+k}⋯M_{c+1}, costs k−1 products per step. After that each leg costs one product per jump, not per layer.
  - The D pairings of old sources are computed only at checkpoint layers. Sources that become old within a step take prefixes of the same J (accumulated backward from c+k).
  - At the intermediate layers the old tier's contribution to (t, D) is given by a cheap rule. First option: pushing MST along the repeated indices, about 3 products per layer. Second: the amplitude form with coefficients read off exactly at the last checkpoint layer.
- **Arithmetic from the k3l atlas (26.27%).** The old tier costs 15.2 points at a0=4 and 18.0 at a0=3; with stride k it divides by roughly k. Including the J chain and MST, this gives:

  | (a0, k) | util | vs atlas |
  |---|---|---|
  | (4, 2) | ≈20.7% | ×0.79 |
  | (4, 3) | ≈18.9% | ×0.72 |
  | (3, 3) | ≈17.0% | ×0.65 |

- **What this changes for Prompt A.** The number f_old stops being a binary verdict. MST without resets requires the loss not to accumulate over the 5–12 remaining layers of the tail's life. With a reset every k layers the error lives at most k−1 steps. The §58 budget (3–5% relative error of the tail) then gives a threshold on f roughly 5–10 times softer.
- **Link to M4.** If the "sliding drop" of M4 assumed layer-by-layer transport for the sake of exact slices, it saves nothing. The synthesis has no jump mechanism.
- **An arm that does not depend on f.** The variant "t exact, D by stride": only the pairings are strided, and only MST is needed for Δ. Estimate about ×0.92.
- **Falsifying test.**
  - A switch `stride=(a0, k, rule)` in `<repo>/experiments/2026-09-24_k3-chain/chain.py`. The legs are transported exactly, as now (in research code this is equivalent to a jump). At non-checkpoint layers the old sources' contribution to t, D, E and the feed is replaced by the rule.
  - Self-check: at k=1 it must give 2.3462e-08 bit for bit.
  - Arms: (4,2), (4,3), (3,3) × {mst, amp}, plus "t exact, D by stride". 10 MLPs, about 65 s per arm.
  - Kill: the best rule at (4,2) is worse than ×1.05 on raw.
  - Pass: ≤ ×1.03. Then compute it in `cost.py` together with the J chain, the MST products and residual.

**2. The wall veto on Strassen L5 depends on the count, and the leaders work in exactly the L5 regime**

- **Idea.** §56 closed L5, but wall is proportional to the count divided by throughput. Throughput in billed FLOPs on the grader (util × 2^41 / wall predict) from the 2026-09-24 telemetry:

  | who | GFLOP/s |
  |---|---|
  | us, k3m (60.9 s) | 8.97 |
  | Luna | 8.97 (the profile of our L4) |
  | J2W | 3.0 |
  | suliman | 3.7 |
  | marius | 4.2 |
  | a_s6 | 3.0 |
  | our L5 projection (23.067% / 119 s) | 4.3 |

  - This is an inference, not a verified fact: it looks like four of the five leaders run one recursion level deeper than us (or with a heavy elementwise layer, which for wall is the same thing). They fit into 77–93 s only because their count is half of ours.
  - For us L5 (×0.9165 measured, accuracy unchanged) becomes admissible when the worst wall 147 s × util/23.07% ≤ 110 s, i.e. at util ≲ 17%. That is exactly where colcut, M2/M3 and gap 1 lead.
  - This is a multiplicative lever that opens by itself as the count falls. The port should be built with the recursion level as a parameter.
- **Falsifying test.**
  - After the colcut port, measure L5 locally at the new count: wall, RSS, residual. Extrapolate linearly from two count points to util 17%.
  - Kill: local wall × 3.9 > 110 s or RSS > 7.5 GB at util ≤ 17%.
  - Separately for the hypothesis "leaders = L5": if their throughput on fresh telemetry is ≥ 7 GFLOP/s, the hypothesis is false.

**3. Per-MLP telemetry as a tool, not just a level**

- **Which assumption is tested.** The leaders' advantage is either the same content transported more cheaply, or "a different method" with no detail on how it differs.
- **What the 50 public MLPs showed** (332142 against the leaders, data already extracted):
  - corr(log(ours/theirs), log theirs): −0.05 for J2W, −0.03 for marius, +0.04 for suliman, +0.07 for Luna.
  - corr(log(ours/theirs), log ours): 0.61–0.70.
  - Residual of our error after regression on the leaders' common factor: sd = 0.079 (R² 0.67). For the leaders themselves, each excluded from the factor: 0.044–0.057.
- **Conclusion.** Our error is their error multiplied by an independent per-MLP excess that does not correlate with how hard the MLP is for them. So we have an extra error channel with its own per-MLP signature, not a uniform difference in quality.
- **Idea.** Use our own structurally different submissions on the same hidden MLPs as a basis of "fingerprints": 332128 (k3f), 332142 (k3h), 332170 (k3i), 332247 (k3j), 332403 (k3m), 332872 (k3n).
  - For each version compute the excess e_m.
  - Regress the excess of k3m on the difference vectors between versions: paths, deadcut, K31/feed, plus Strassen as a control, which should give about 0.
  - The strongest correlate will point to the channel the leaders carry correctly.
- **Second tool.** Regress each leader's per-MLP FLOPs on ours (our count depends on the number of alive neurons through deadcut).
  - r > 0.8: they cut the same dead neurons in O(n³) contractions; that is the signature of our family. The slope shows the share of their count in such contractions.
  - r near 0: their count is not about source legs.
- **Falsifying test.** The tool is exhausted if k3m's residual after regression on the leaders' common factor has sd ≤ 0.057, or no difference vector correlates with it at |r| > 0.3 (p ≈ 0.03 at n = 50). About an hour of work, no compute.

**4. The soft channel: cutoff thresholds by source age**

- **Which assumption is tested.** deadcut, colcut, M2 and M3 use one threshold for all sources. Meanwhile §58 showed that the old tail is a soft channel: a budget of 3–5% relative error, no √n discount. The young legs and the variance are hard: a parabola of width 1e-5, leg A is the sensitive one.
- **Why it is worth testing.** Layers 10–15 hold 46% of the count. At layer 15, 18% of neurons have α < −3 (MLP 0). By §2, 58% of the neurons there lie outside |α| < 2.
- **Idea.**
  - Cut the input rows only for old legs, with θ_old ∈ {−2.0, −1.5}. The young stay at −2.5; they can even be tightened to −3.0 at the same price.
  - M3 (receiver rows) only for old legs and not at the last layer, where the t_i of dead neurons is worth about 4e-9.
  - The dropped terms are multiplied by the gate d1 ≤ Φ(θ_old). So the readout-weighted error is much smaller than in the plain norm.
  - This is not a repeat of M3: there it is one threshold θ_out ≤ −3.0 for all ages.
- **Ceiling.** Plus 5–12% of cut rows in the old tier (15 points); the alive×alive pairings give a quadratic effect. Total about 1.5–3 points (×0.88–0.94).
- **Falsifying test.**
  - Split the mask in `chain.py` by age (a0=4). Arms θ_old ∈ {−2.0, −1.5, −1.0}, 10 MLPs.
  - Kill: the cliff for the old legs alone sits where it does for the single threshold (a single −2.0 gave ×1.215), i.e. θ_old = −2.0 gives ≥ ×1.10 on raw.
  - Pass: θ_old = −1.5 gives ≤ ×1.03 with a saving of at least 1.5 points.

**5. The rule "nothing below 3% ships" does not fit cost levers**

- **Which assumption is tested.** Both the brief and §3 of the synthesis assume that the 0.4–0.5% local→grader discrepancy makes any lever below 2–3% unverifiable.
- **Why it is wrong for cost.**
  - adjusted = raw × util, and the discrepancy was measured on raw (§60, the k3n scalars).
  - flopscope counts FLOPs analytically, identically locally and on the grader. For a change that does not touch accuracy, the util ratio transfers exactly.
  - For data-dependent cuts, the spread comes from the distribution of MLPs, not from a transfer error.
- **The rule worth applying.** For M5 (×0.983, deterministic), for the cost part of colcut, M2, M3 and for small exact levers:

  exact util ratio × upper bound of the raw ratio (paired test on 100 local MLPs, SE ≈ 0.1%) + 0.5% for transfer < 1.

  Three levers of 1.5% each together give a guaranteed ~4.5%. Under the current rule all three are thrown out.
- **Falsifying test.** Compare the util ratio of two graded submissions with different FLOP logic (332247 k3j against 332403 k3m) with the same ratio on 100 local MLPs (k3j 26.27%, k3m 24.894% locally). If the discrepancy is more than 0.3%, my reading is wrong and the threshold stays.