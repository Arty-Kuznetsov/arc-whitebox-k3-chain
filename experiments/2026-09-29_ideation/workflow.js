export const meta = {
  name: 'arc-first-place-ideation',
  description: 'Seven independent research lenses propose overlooked paths to #1, each vetted adversarially against 61 negative results and the rules, then ranked',
  whenToUse: 'When a research project has exhausted its obvious levers and needs genuinely new directions checked against a catalogue of dead ends',
  phases: [
    { title: 'Ideate', detail: 'seven researcher lenses, each proposing concrete ideas with priced construction cost' },
    { title: 'Vet', detail: 'one adversarial skeptic per lens batch: already closed? allowed? construction priced?' },
    { title: 'Synthesize', detail: 'dedup, rank by expected score gain x probability, test plan' },
    { title: 'Critic', detail: 'completeness critic: which idea classes were never explored' },
  ],
}

const ROOT = '<repo>'

const CONTEXT = `
You are a senior researcher joining the ARC White-Box Estimation Challenge 2026 project at ${ROOT}.
Read-only: DO NOT edit any file, DO NOT run long computations, DO NOT submit anything anywhere.

THE TASK. Given weights of a random He-init ReLU MLP (width n=1024, depth 16, y=ReLU(W^T x), W~N(0,2/n), input N(0,I)),
predict per-neuron mean post-ReLU activation for all layers. Score = final_layer_mse x max(0.1, C/B), B=2^41 FLOPs
counted analytically by the flopscope library. Multiplier FLOORS at 0.1: below 10% of budget FLOPs are free.
All arithmetic must go through flopscope.numpy; shipping data files is allowed. Hard caps: wall 120 s, residual
(non-flopscope python) time 0.4 s, RSS 8 GB, per MLP.

WHERE WE ARE (all measured).
- Our best graded: 5.3906e-9 = raw 2.1652e-8 x util 24.835%. Method: ARC's reference k=3 cumulant chain (Wu et al.)
  re-derived, plus our scale-mixture kappa4 closure, the C0.D diagram, a kappa4->kappa3 feed, a dead-ROW cut
  (neurons with alpha=mu/sigma < -2.5 gated to 0 and dropped from contractions), Strassen level 4 (billing 0.636).
- Board: #1 J2W 1.70e-9 (raw 1.62e-8, util 10.5%), #2 1.90e-9 (1.42e-8, 13.4%), #5 2.20e-9. Thirteen entrants sit at
  util 10.5-18%. We are the cost outlier: our raw is within x1.5 of theirs, our util is x1.6-2.4 worse.
- Cost atlas of our chain: leg transports 12.79% of B, D pairings 10.62%, three base terms ~1% each. One kappa3
  "source" is born per layer (legs A_s, B_s are n x n, A born as identity, B = diag(d1) C0; readout
  t_i += 3 sum_p A[i,p] g_p B[i,p]^2 with g = d2 at birth) and every source is transported by W^T every layer:
  2 n^3 per source per layer, O(L^2) total. If everything of source age >= a were free: a=2 5.3%, a=3 8.3%,
  a=4 11.1%, a=5 13.6%, a=6 15.9%.
- With cost free (at the floor) our accuracy stack reaches raw ~1.72-1.92e-8 -> 1.72-1.92e-9, i.e. rank 1-2. NONE of it
  ships at 24.8%. So first place is a COST problem.
- The grader ranks on a mean over 50 hidden MLPs; measured local->grader discrepancy ~0.4-0.5%, so any lever worth
  <2-3% cannot be validated. Score prizes are decided by a private re-evaluation on fresh MLPs.

WHAT IS CLOSED (read ${ROOT}/notes/NEGATIVE-RESULTS.md before proposing; sections 54-61 are the recent ones):
- The unifying mechanism of 54, 55, 56, 58, 59: "a cost model that prices the product a compression saves but not
  the CONSTRUCTION the compression needs says yes every time". Shared subspace (entry update ate it), low-rank path
  leg (basis costs 4q), Strassen L5 (memory bandwidth, wall x1.95), fitted rank-n CP block for the old tail (block
  EXISTS but one fitting pass = the whole per-layer saving, CP optimum not nested, no warm start), unbiased importance
  sampling of the tail's CP terms (kappa = sum||term||/||T|| = 62, i.e. 194% error; the tail's terms are same-sign in
  the diagonal readout but nearly ORTHOGONAL in Frobenius norm).
- 57: coherence. The readout sums same-sign terms over n neurons: a correction SMOOTH across neurons is paid in full,
  an incoherent one is discounted by sqrt(n). The chain sits in a variance-level parabola 1e-5 wide; every smooth fit
  of the residual is worse than nothing. The final-layer residual is 99.5% incoherent and uncorrelated (|corr|<=0.03)
  with every local feature, and corr(residual layer15, residual layer14)=0.004 (each layer's random W scrambles it).
- 60: scalar tuning is below the grader's resolution. 61: every live accuracy lever costs more than it gives at 24.8%.
- Also closed: window truncation (16), Tucker (18), low-rank B/D (20,21; but note: the OLD tail's D alone is ~rank 256
  at x1.008), per-source low rank (22), amplitude closure of the tail (39), aug diagrams (x0.46), e3 (x0.94),
  pure Monte-Carlo (200x worse than us at full budget).
- NEW, measured 2026-09-29 (10 MLPs, shippable base): NEGATIVE 48 revisited. Dropping birth COLUMNS p (parent
  neurons) splits sharply by side. DEAD parents (alpha_p < -2.5): raw x0.9928, i.e. slightly BETTER on 10/10 MLPs,
  6.4% of columns dropped cost-weighted -> projected x0.936 on the score; a cliff below -2.0 (x1.40 at -1.5).
  SATURATED parents (alpha_p > +3.0): raw x1.12 although they carry only 0.03% of the d2-weighted birth mass --
  a saturated neuron is ~identity, so kappa3 passes THROUGH it and its column carries transported content, not a
  birth. So 48's premise ("all hub weights ~0 in the linear regime") holds only on the dead side. The dead-side
  column cut is now being ported to production on the Mac; do not re-propose it, but DO build on the mechanism
  (what else is carried by columns/rows that carry no information?).
- Judgments of the form "not worth it" made before 28 Sep were made when bigger levers existed; they are not
  accuracy verdicts and may be wrong now.

AN OPEN TEST RUNNING ELSEWHERE (do not duplicate): a cloud session is testing whether (t, D, C, kappa4) is a
SUFFICIENT STATE for kappa3 (i.e. can a block fitted only to the two slices the nonlinear step reads carry the chain).

Useful deeper files: ${ROOT}/CLAUDE.md (CURRENT STATE block), ${ROOT}/notes/FRONTIER.md (top blocks),
${ROOT}/notes/METHOD.md (Part II = the chain), ${ROOT}/experiments/2026-09-26_cost-frontier/RESULTS.md (the atlas),
${ROOT}/src/estimator_k3m.py (the shipped estimator), ${ROOT}/experiments/2026-09-24_k3-chain/chain.py (torch research
chain with every switch), ${ROOT}/external/flopscope/docs/reference/cost-model.md (how every op is billed),
${ROOT}/notes/SPEC.md (rules: section 5.5 fair accounting, section 6 source authenticity).
`

const LENSES = [
  { key: 'billing-algos', title: 'Fast algorithms and the flopscope billing rules',
    brief: `Study how flopscope bills every op (cost-model.md: the contraction family, the symmetry-aware engine that
bills inner(A,A) at half, einsum accumulation, batched ops, free ops, dtype rates). Find LEGITIMATE reductions
(rules 5.5: symmetry, better algorithms, lower precision are encouraged; mis-declaring or bit-packing is not).
Where in OUR chain could a product be restructured so the engine can prove a symmetric output, or be replaced by a
cheaper-billed op computing the same thing? Consider the D pairings sum_s (L1_s B_s^T + L2_s A_s^T), C_pre = W^T C W,
the leg transports, and any Gram-like products. Price everything including extra ops.` },
  { key: 'rmt', title: 'Random-matrix theory, mean-field and free probability',
    brief: `W is He-random and KNOWN, but its statistics are universal. Where can a deterministic equivalent,
concentration, or a self-averaging argument replace an n^3 transport by O(n^2) work plus a small correction? Think
about what the leg transports W^T A_s actually need to deliver (only row-wise triple products and the D pairing).
Think about the old tail specifically: after many random transports, does it become 'generic' in a way that admits a
closed-form statistical description? Be concrete about which quantity is replaced and how its error would be measured.` },
  { key: 'restructure', title: 'Algebraic restructuring of the cumulant chain itself',
    brief: `Read estimator_k3m.py and METHOD Part II. Look for redundancy in the algebra: A_s is the transport of the
identity (A_s = M_s = W_l^T D1 ... W_{s+1}^T), so all A legs are suffix products of ONE chain; B_s = M_s B_s^0.
Can contractions be reordered or shared across sources? The dead-side birth-column cut just worked (see context) and
the saturated side failed because kappa3 passes through near-linear neurons. Building on that mechanism: which other
rows, columns, slots or whole contractions carry negligible information and are still transported every layer? Which
layers or source ages could use a cheaper variant (the first readout, the last layer, the oldest sources)? Price every idea.` },
  { key: 'tensor-formats', title: 'Tensor formats beyond CP and Tucker',
    brief: `CP compression of the tail exists but cannot be constructed cheaply and is not nested; Tucker/projection
compressions are closed. Consider hierarchical, tensor-train, block-structured, symmetric-sum-of-cubes, or streaming/
incremental formats where adding one source is O(n^2) or O(n^3) exactly (NESTED by construction), or where the
construction is a by-product of computations the chain already does. The deciding constraint is: construction cost per
layer must be <= ~1 dense n^3 product while keeping the tail's relative error <= 3-5%. Be explicit about nestedness.` },
  { key: 'stat-sketch', title: 'Statistics, sketching and hybrid estimators',
    brief: `NEGATIVE 59 closed unbiased sampling of the tail's CP terms because of the FROBENIUS norm. But what if the
object sampled or sketched is not the tensor but the READOUT (t and D) directly, where coherence is favourable? E.g.
estimate t_old = sum over old sources and birth columns by a sketch whose variance is governed by the readout norm, or
use control variates built from the young sources, or randomised trace/diagonal estimators (Hutchinson/Hutch++ style)
for the diagonal contractions the readout needs. Consider hybrids with a small Monte-Carlo correction on top of the
analytic chain (budget: 10% of B is ~6500 forward samples). Quantify variance honestly.` },
  { key: 'learned', title: 'Learned closures and surrogates within the rules',
    brief: `Shipping data files is allowed; ARC prefers mechanism over fitted constants but does not forbid learned
components. Our kappa4 IS already closed in O(n^2) by a scale-mixture ansatz. Could a kappa3 closure be LEARNED
offline (coefficients of a small set of O(n^2)-state diagrams fitted on many random MLPs, the full chain as teacher)?
Respect NEGATIVE 57 (smooth corrections act coherently) and 60 (fitted scalars do not transfer below ~2-3%). What
would have to be true for a learned closure to generalise to fresh MLPs, and how would you test it cheaply?` },
  { key: 'intel', title: 'Competitive intelligence and reasoning backwards from the leaders',
    brief: `Thirteen entrants sit at util 10.5-18% with raw 1.42-2.16e-8. Reason BACKWARDS from that telemetry: what
representations of the third cumulant (or alternatives to it) have a per-layer cost of ~8-12 dense products, constant
in depth, and could reach raw ~1.6e-8? Also try to read the public discussion forum of the challenge on AIcrowd
(https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/discussion) and ARC's own published
material for hints about what works; use WebFetch/WebSearch via ToolSearch if available. Report only what you actually
read, with URLs; never invent forum content. Note: code derived from another participant's submission is disallowed
(rules 6), but public ideas and papers are fair game with attribution.` },
]

const IDEA_SCHEMA = {
  type: 'object',
  properties: {
    ideas: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          mechanism: { type: 'string', description: 'why it could work, concretely, which quantity changes' },
          cost_estimate: { type: 'string', description: 'per-layer cost in dense n^3 units INCLUDING construction; projected util' },
          accuracy_risk: { type: 'string' },
          closest_negative: { type: 'string', description: 'the NEGATIVE-RESULTS section it is closest to, and why it differs' },
          cheap_test: { type: 'string', description: 'the cheapest measurement that would falsify it' },
          projected_adjusted: { type: 'string', description: 'projected adjusted score if it works, with the arithmetic' },
        },
        required: ['title', 'mechanism', 'cost_estimate', 'accuracy_risk', 'closest_negative', 'cheap_test', 'projected_adjusted'],
      },
    },
    sources_read: { type: 'array', items: { type: 'string' }, description: 'files/URLs actually read' },
  },
  required: ['ideas', 'sources_read'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          already_closed_by: { type: 'string', description: 'NEGATIVE section that closes it, or "none"' },
          allowed_by_rules: { type: 'boolean' },
          construction_cost_priced: { type: 'boolean' },
          revised_cost_and_projection: { type: 'string' },
          verdict: { type: 'string', enum: ['promising', 'marginal', 'closed', 'disallowed'] },
          reason: { type: 'string' },
        },
        required: ['title', 'already_closed_by', 'allowed_by_rules', 'construction_cost_priced', 'revised_cost_and_projection', 'verdict', 'reason'],
      },
    },
  },
  required: ['verdicts'],
}

const results = await pipeline(
  LENSES,
  (lens) => agent(
    `${CONTEXT}\n\nYOUR LENS: ${lens.title}.\n${lens.brief}\n\nPropose 3-5 CONCRETE ideas that could move us toward first place. ` +
    `Quality over quantity: each idea must name the exact quantity it changes, price its construction cost per layer, ` +
    `state the NEGATIVE-RESULTS section it is closest to and why it is different, and give the cheapest falsifying test. ` +
    `Discard ideas that are restatements of closed ones. It is fine to return fewer ideas if that is all that survives your own scrutiny.`,
    { label: `ideate:${lens.key}`, phase: 'Ideate', schema: IDEA_SCHEMA }
  ),
  (ideated, lens) => {
    if (!ideated || !ideated.ideas || !ideated.ideas.length) return null
    return agent(
      `${CONTEXT}\n\nYou are an ADVERSARIAL REVIEWER. A colleague using the lens "${lens.title}" proposed the ideas below. ` +
      `For EACH idea: (1) search ${ROOT}/notes/NEGATIVE-RESULTS.md for a section that already closes it -- if one does, ` +
      `say which and verdict "closed" (default to closed when a section clearly covers the same mechanism); (2) check it ` +
      `against the rules in ${ROOT}/notes/SPEC.md (fair accounting 5.5, source authenticity 6); (3) check whether its ` +
      `cost estimate includes the CONSTRUCTION cost, and re-price it yourself if not -- this is the failure mode of five ` +
      `past ideas; (4) give a revised projection. Only mark "promising" if it survives all four and would be worth >= 3% ` +
      `on the adjusted score (below that the grader cannot see it).\n\nIDEAS:\n${JSON.stringify(ideated.ideas, null, 2)}`,
      { label: `vet:${lens.key}`, phase: 'Vet', schema: VERDICT_SCHEMA }
    ).then(v => ({ lens: lens.key, title: lens.title, ideas: ideated.ideas, sources: ideated.sources_read, verdicts: v ? v.verdicts : [] }))
  },
)

const batches = results.filter(Boolean)
log(`${batches.length}/${LENSES.length} lenses produced vetted batches; ` +
    `${batches.reduce((a, b) => a + b.ideas.length, 0)} ideas, ` +
    `${batches.reduce((a, b) => a + b.verdicts.filter(v => v.verdict === 'promising').length, 0)} marked promising by their reviewer`)
if (batches.length < LENSES.length) {
  log(`WARNING: ${LENSES.length - batches.length} lens(es) returned nothing -- the synthesis below does not cover them`)
}

phase('Synthesize')
const synthesis = await agent(
  `${CONTEXT}\n\nYou are the research lead. Below are ideas from independent lenses, each already reviewed by an ` +
  `adversarial skeptic. Produce the decision document:\n` +
  `1. Deduplicate across lenses (the same mechanism proposed twice counts once; note which lenses converged -- ` +
  `convergence is evidence).\n2. Rank the surviving ideas by (projected gain on the adjusted score) x (your honest probability ` +
  `it works), and show that arithmetic.\n3. For the top 3-5: the exact first experiment, its cost in wall time on a MacBook ` +
  `(torch chain ~6 s per MLP per config at 10 MLPs) or in a cloud session, and the number that would kill it.\n` +
  `4. A short, honest bottom line: given 18 days to the 17 Oct submission close, what is the realistic probability of ` +
  `first place, of top-5, and what is the single highest-value next action.\nBe skeptical; do not inflate. ` +
  `Where the skeptic and the proposer disagree, say who you believe and why.\n\nBATCHES:\n${JSON.stringify(batches, null, 2)}`,
  { label: 'synthesize', phase: 'Synthesize' }
)

phase('Critic')
const critique = await agent(
  `${CONTEXT}\n\nYou are a completeness critic reviewing a research ideation exercise. Seven lenses were run: ` +
  `${LENSES.map(l => l.title).join('; ')}. Their synthesis is below. What CLASS of idea was never considered by any lens? ` +
  `What assumption do all seven share that might be wrong? Is there any reading of the task, the metric, or the rules ` +
  `that everyone has taken for granted? Name at most 5 gaps, each with a one-paragraph concrete idea and a falsifying test. ` +
  `Do not repeat ideas already in the synthesis.\n\nSYNTHESIS:\n${synthesis}`,
  { label: 'completeness-critic', phase: 'Critic' }
)

return { synthesis, critique, batches }
