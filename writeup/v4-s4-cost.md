## 4. Cost: where the budget goes, and what binds it

The score is `mean_m( final_layer_mse_m × max(0.1, C_m/B) )` with `B = 2⁴¹` FLOPs per network as billed by flopscope.
Two features of that formula shaped every decision in §3.

**It is a product, so relative percentages are interchangeable.** At our operating point 1% of utilisation and 1% of
MSE are worth exactly the same, which is why every lever in this document is reported as a pair (raw ×, util ×) and why
several terms that *improve accuracy* are absent from the shipped estimator: they improve it by less than they cost.

**The multiplier floors at 0.1.** Above 10% of the budget every FLOP is charged linearly; at or below it FLOPs are free
and only accuracy matters. We never got under the floor (k3z: 18.33%). At our raw error, the floor alone would have
been worth ×0.55: 2.03e-9, 16th place on the final board of 21 October, and §6.1–6.2 are the record of
why it could not be reached with this representation.

### 4.1 The measured cost map of the shipped estimator

We stopped pricing levers with a cost model after it was wrong by 3.5×, by 9× and in sign within one day (§6.1 records
what that model said yes to). The map below is the meter itself: the shipped estimator with a `flopscope.namespace`
around every call site (a copy verified bitwise against the shipped file), run through the official runner on one
public network [M]:

| call site | what it is | FLOPs | backend time |
|---|---|---|---|
| leg transport | the young sources' legs and the band basis Pᵀ moved one layer (Strassen level 5) | **23.5%** | 19.3% |
| D pairing | the (2,1) slice from the non-members' legs (level 5) | **19.9%** | 20.9% |
| band members | the old sources' legs materialised from the basis, for the readout (level 5) | **16.7%** | **24.2%** |
| members' pairing | the old sources' part of D, paired in the compressed space (level 4) | 11.8% | 10.6% |
| band entry | QR of the basis, projection, explicit triangular inverse | 8.2% | 4.4% |
| covariance | C and the newest path leg transported (level 5), C_pre = Wᵀ(CW) (3 of 4 blocks, level 5) | 11.5% | 13.4% |
| the step's mathematics | the whole pair expansion **0.43%**, the κ₄ closure 0.06%, its learned corrector 0.05%, the readout 1.9%, the births 0.9% | **3.3%** | 2.9% |
| the rest | pairing preparation, thin legs, the E2 level restoration, gathers and gating | 5.1% | 4.3% |

<figure><img src="figures/fig_v4_costmap.png" alt="Figure 2"><figcaption>Figure 2. The table as a picture: share of the designated estimator's FLOPs (dark) and of flopscope's backend time (light) by call site, one network, <code>flopscope.namespace()</code> on a bitwise copy of k3z.</figcaption></figure>

**Four fifths of the budget is the logistics of carrying the third cumulant forward; the mathematics that decides the
answer is 3%.** Two consequences follow, and both shaped the work. Any accuracy term that lives in the step, such as a new pair diagram, a closure pattern or the learned corrector, is free, which is why §3.2–§3.5 and §3.7 cost nothing
measurable. And any real cost saving has to come out of the representation of κ₃, which §6.1 and §6.2 show is the one
place no cheap construction exists. The level-5 sites run at 8–14 billed GFLOP/s against 17–30 for level-4 and
element-wise work: deeper Strassen recursion buys FLOPs with wall time.

### 4.2 The configuration sits at the kink of its own cost/accuracy curve

Eight structural knobs were measured in both directions on ten networks: the band rank and entry age, re-opening all
source ages for the path births, a larger thin rank, the κ₄ → κ₃ coupling, and others. The elasticity
|d log raw / d log util| is **below 1 in every direction that spends more** (band rank 512: 0.43; entry age 6: 0.56;
all ages: 0.25; thin rank 384: 0.10; κ₄κ₃: 0.05) and **above 1 in every direction that spends less** (rank 384: 1.54;
entry age 4: 3.76) [M]. The product raw × util is minimised exactly where the estimator ships; a fifty-three-switch
catalogue re-screened by a cloud session found no switch that improves it either (12 of 53 improve raw, none the score).

### 4.3 The budgets that are not the meter

Three caps zero a network's prediction when crossed (400 ms of non-flopscope time, 120 s of wall time, 8 GB of memory),
and in practice they, not the FLOP count, set the depth of every engineering choice. Their measurements, the Strassen
depth they allow and the designation they imply are in Appendix A.2–A.3.
