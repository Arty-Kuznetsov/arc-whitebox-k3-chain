# The cost map of the shipped estimator k3y (2026-10-02) — FLOPs AND backend time per call site

`instrument.py` builds `estimator_k3y_prof.py` = `src/estimator_k3y.py` with `flopscope.namespace()` enter/exit lines
around every call site (no re-indentation); `check_prof.py`: the copy is **bitwise k3y with equal FLOPs** (4.095564e11
on the synthetic 1024×16). One public network through the official runner (`whest run --n-mlps 1`, mini MLP 0,
`prof_k3y_n1.json`): util **18.82%**, raw 2.1923e-8, wall 37.9 s (backend 35.2 s, overhead 2.3 s), residual 0.459 s
(local, contended machine), 23 912 flopscope calls.

| call site | what it is | FLOPs share | backend time share | calls | billed GFLOP/s |
|---|---|---|---|---|---|
| `transport` | the young sources' legs and the band basis Pᵀ moved one layer (level 5) | **23.5%** | 20.5% | 1695 | 13.5 |
| `dpair` | the D pairing over the non-members' hub rows (level 5) | **19.8%** | 21.6% | 1624 | 10.8 |
| `band_members` | the members' legs materialised, [Z_Bᵀ; Z_Aᵀ]Pᵀ (level 5) | **16.8%** | **25.2%** | 1630 | 7.8 |
| `mpair` | the members' D pairing in the compressed space (level 4) | 11.7% | 10.8% | 2110 | 12.7 |
| `ctransport` | C and the newest path leg moved with the full W (level 5) | 8.3% | 7.8% | 1695 | 12.5 |
| `band_entry` | QR of P, projection, explicit R⁻¹ (binv) | 8.2% | 4.5% | 2155 | 21.6 |
| `cpre` | C_pre = Wᵀ(CW), 3 of 4 blocks (csym) | 3.4% | 2.4% | 2520 | 16.8 |
| (unnamespaced) | E2 level restoration, gathers, gating, elementwise | 2.8% | 3.7% | 2278 | 8.9 |
| `readout` | t from the legs | 1.9% | 0.8% | 933 | 26.5 |
| `pairprep` | L1, L2 | 1.6% | 0.6% | 651 | 30.4 |
| `births` | the new source: B, the range finder, H | 0.9% | 0.5% | 701 | 20.2 |
| `thin_transport` | the thin legs | 0.7% | 0.0% | 15 | 488 |
| `nonlin` | **the whole pair expansion** (Q11, Q21, Q22, Leonov–Shiryaev) | **0.43%** | 1.3% | 4370 | 3.8 |
| `k4levels` | the κ₄ closure's levels | 0.06% | 0.2% | 755 | 4.5 |
| `k4corr` | the learned κ₄ corrector | 0.05% | 0.1% | 780 | 5.2 |

**Moving the κ₃ representation (leg transport, the members, both D pairings, the band entry) is 80% of the FLOPs and
83% of the backend time; the covariance itself (C and the path leg transported, C_pre) is 11.7% / 10.2%; the
mathematics of the step — the pair expansion, the κ₄ closure and its corrector, the readout and the births — is
3.3% of the FLOPs.** The level-5 sites run at 8–14 billed GFLOP/s against 17–30 for
level-4 and elementwise work: Strassen's deeper levels buy FLOPs with wall, which is why §69/§6.1's wall budget, not the
meter, ended that axis. Compared with k3r's map (`notes/COST-MAP-2026-10-01.md`, a different network): the members'
compressed pairing, which was hidden in the unnamespaced block (estimated ≈7%), is 11.7% measured.

**Peak memory** (`rss_k3y.py` under `/usr/bin/time -l`, one predict on the synthetic 1024×16): **maximum resident set
6.91 GB of the 8 GB cap.** Most of it is the level-5 Strassen workspace (needed as soon as one site runs at level 5);
k3p, before any level 5, peaked at 4.5 GB, and level 5 everywhere at 7.21 GB. Every graded k3t…k3y run had 0 failures
on all 100 networks, so the grader's accounting is inside the cap — but the margin is ≈1.1 GB, worth knowing before
any change that allocates more.


## k3z (2026-10-02 evening) — the same map for the next designated candidate
`python instrument.py k3z` (the anchors are unchanged) → `estimator_k3z_prof.py`; `check_prof.py k3z`: **bitwise k3z,
FLOPs equal** (4.070984e11 on the synthetic 1024×16). One public network (`prof_k3z_n1.json`): util 18.712%, wall
37.8 s, residual 0.427 s (contended), 23 918 calls. Shares (FLOPs / backend time): transport 23.5 / 19.3, dpair
19.9 / 20.9, band_members 16.7 / 24.2, mpair 11.8 / 10.6, band_entry 8.2 / 4.4, ctransport 8.3 / 7.2, **cpre 3.2 / 6.2**
(level 5 inside the halves: slower per FLOP), nonlin 0.4 / 1.3.
