# A third-cumulant chain at its boundary: consistency, random projections, the price of carrying κ₃, and where the next order's accuracy lives

*ARC White-Box Estimation Challenge 2026 — Phase 2 — Algorithmic Contribution write-up.*
*Participant: **Artyom Kuznetsov** (AIcrowd `Artyom_Mazur`). Designated submission: **333534**, graded
**3.7357e-9** (final-layer MSE 2.0317e-8 at 18.33% of the FLOP budget; 0 failures on all 100 grader networks).*
*Claim labels: **[M]** measured in a logged run · **[D]** derived algebraically · **[U]** our interpretation, not isolated
by an intervention. The use of language models in this work is described in §9.*
*Revised 27 October 2026, after the close: board numbers of the close (§4, §6.2, §7, §8), the release links (§10), one
new measurement in §6.4 (the error along the mean) and later forum work that extends these results (§6.4, §7). The
version posted on 16 October is otherwise unchanged.*

---

## Summary

**Task and result.** Predict the per-neuron mean post-ReLU activation of every layer of a random He-initialised ReLU
network (width 1024, depth 16, Gaussian input) within 2⁴¹ analytical FLOPs; the score is the final-layer MSE times
max(0.1, FLOPs/budget). Our estimator is a cumulant-propagation chain closed at the third cumulant. Its skeleton is the
SIMPLE k = 3 algorithm of Wu et al. (arXiv:2605.05179) and their MIT reference implementation, re-derived and
re-implemented from scratch (§3.1); everything else in §3 is ours unless a credit says otherwise. The designated
submission scores 3.7357e-9: ×108 better than covariance propagation, ×2.1 more accurate than the reference algorithm at
×8 less compute. It is the last of ten consecutive versions that
each landed within 0.3% of their pre-submission projection (§5.1).

**What we would like judged.** Six measured statements about why this family of estimators behaves as it does. Each
is stated as a rule with the prediction it makes, and each is backed by an oracle-substitution or a metered measurement
rather than by an argument.

1. **The next layer's per-neuron errors are random projections of this layer's joint errors** (§6.4, Proposition 1).
   ΔC(l), the error of the chain's post-activation covariance, is independent of W(l+1), so the variance error of
   neuron i is wᵢᵀΔC wᵢ: a coherent level from tr ΔC plus independent noise across neurons set by the off-diagonal
   part. Verified against 10⁷-sample covariance oracles (predicted/measured 0.89–1.09; 95% of the energy is
   off-diagonal). The κ₃ error is an odd-order projection and has no coherent level at all (Proposition 2; z² = 0.93
   over 28 layer–network pairs), and the real columns' errors are spread exactly like the chain's errors along random
   directions independent of the network (spread ratios 0.96–1.12 and equal means over four layers of two networks). This is why every corrector of a neuron's moments from its own state failed on our
   chain: the part of the error such a model can see is worth ×1.0; the part it cannot is worth ×0.3. What a per-neuron
   correction can learn is set by the truncation order, not by depth: about two thirds of a Gaussian closure's error at
   every shape from 256 × 8 to 1024 × 16, ≤ 4% of a k = 3 chain's, because a neuron's true κ₃ is proportional to its
   own mean: the slope at every layer is the projection of the post-activation partial trace on the mean (exact in
   expectation, confirmed to ~1%), its first value is derived (c(1)·n = 13.91, measured 13.94), and carried inside a
   Gaussian closure the law removes exactly the learnable part (§6.4). `scripts/error_anatomy.py` measures this for any estimator from its outputs.
2. **The third cumulant is not closed on the two slices the nonlinear step reads** (§6.2). The all-distinct part of κ₃
   is 28.8% of the next layer's diagonal per layer, orthogonal to everything computable in O(n²), and ×25.6 end to end
   without it. κ₄ can be closed in O(n²) because it enters the readout at second order; κ₃ is the readout. The 28.8% is
   derived, not only measured (Proposition 3: share² = 6‖G<sub>AD</sub>‖² / (6‖G‖² + 9‖v‖²), measured/predicted 1.00 ± 0.06 at
   n = 256), and it falls only slowly with width. Independently replicated by team 504aldo (forum 18218) with different
   code and a different instrument.
3. **The construction costs what it saves** (§6.1). Five compressions of the κ₃ state, each affordable by a cost model
   that priced the saved product, each dead on the meter when the construction of the compressed object was priced:
   the basis, the fit, the rotation, the entry. The one compression that ships (§3.6) is the one whose basis is never
   rebuilt. The rule predicts the fate of compressions we never tried.
4. **Consistency: the chain's errors compensate, so remove a term on both sides or not at all** (§3.5, §2c). Three
   changes that removed computation and gained accuracy (matched dead-neuron cuts, saturated rows of D with the κ₄
   level they carried restored, one Edgeworth coefficient dropped in all three pair channels); and the mirror image
   from the oracle side: exact variance alone can make the result worse (×1.09), exact variance, κ₃ and κ₄ together
   give ×0.12.
5. **One order up: the missing accuracy is the young fourth cumulant, and we can say which trees** (§7). At small width
   ARC's k = 4 is ×3–7 better than our chain; its gain lives in the κ₄ born in the last two layers, above all in the
   graph that hands the whole third cumulant on through one covariance edge (the "3+2" tree; without it the young κ₄ is
   worse than none). Inside ARC's own k = 4, three layers of κ₄ blocks plus one scale-mixture block for the rest keep
   89% of its gain over k = 3 at 256 × 16 for 0.4 of the κ₄ columns (§7.3). A reader of those trees built from our
   chain's own state cuts our raw error to ×0.70 on all ten
   full-width test networks, close to the leaders' raw error; metered in our representation, it costs about what it buys (×0.97 in
   score at best), so it is not in the designated submission. The one variant we did port, to measure it on the
   grader, is ×0.92 raw on the hundred public networks at +5% FLOPs, and the grader zeroed 16 of them for residual
   time: the construction is priced in metered calls as well as in FLOPs (§7.4).
6. **The instrument has a resolution** (§6.5): a lever worth under 2–3% cannot be validated by a submission, and a
   scalar fitted on ten networks arrives at a third of its strength on a hundred. Structural levers transfer; the ten
   ports that landed on their projections are the evidence.

**The method, briefly** (§3; ablation on 32 fresh networks in §5.2). A κ₄ closure in O(n²) from the Gaussian scale
mixture (κ₄ᵢ ∝ varᵢ², K22 ∝ varᵢvarⱼ + 2C0ᵢⱼ²), which pays only together with the edge × κ₃ diagram C0·D that the
reference truncates (together ×1.136; each alone ≤ ×1.07 or harmful; on the final stack ×1.30 and ×1.17 when removed);
the rest of the first-order scale-mixture set, including a κ₄ → κ₃ feed; the augmented path births where they are cheap;
the consistency cuts of item 4; old sources in a rank-448 basis that rides the transport (×0.893, 100/100 networks); one
learned correction of the κ₄ diagonal from O(n²) row reductions of matrices the chain already holds (the idea is
mliston's, forum 18170; ×0.988 out of sample and ×0.988 on the grader); and exact arithmetic — Strassen–Winograd as
flopscope operations, symmetric and triangular savings — which we report in Appendix A because it is engineering.

**What we did not achieve.** The top of the board is ×1.5–1.8 more accurate and ×1.5–1.8 cheaper than this method, by
means that are not published. Item 5 explains the accuracy; the cheaper base the leaders must have for the κ₃ state is
the part we did not find, and §6.1–§6.2 say why none of the constructions we could price gives it. §8 lists what we
would try next.

**How this document was made.** Every number comes from a logged run with a self-check (§1); the journal of the work,
including seventy-six catalogued negative results with their mechanisms, is released with the code (§10). The
participant directed the research and Claude Code wrote the code and the drafts under that direction; §9 states the
division of labour and the errors the model made.
