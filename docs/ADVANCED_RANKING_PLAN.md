# Advanced ranking & dialogue plan — LTR, relevance feedback, VoI clarification

Three standard techniques the audit + literature say a competitive shopping copilot should have and
ours lacks. Each targets a measured weakness, fits the competition constraints (frozen text catalog,
distinct public/private users, no heavy infra, LLM optional), and ships behind a flag with a decision
gate. Priority order: (1) LTR, (2) relevance feedback, (3) VoI clarification.

Context from measurement (current default = satisfaction ranker + cross-encoder):
public 0.891, pillar_moderate ~0.50, pillar_free ~0.66 (honest). Ranking is ~82% of the honest-set
loss; the remaining gap is precision among semantic look-alikes. Cross-encoder helped a lot but costs
public (−0.026) because it is un-gated. LTR is the principled way to combine all signals and to learn
*when* each helps (e.g. down-weight the cross-encoder on verbatim turns → recover public).

---

## 1. Learning-to-Rank (LTR) — the primary gap

### Motivation
We hand-blend many ranking signals with manually-tuned constants (retrieval rank, coverage,
satisfaction, cross-encoder, price, popularity). LTR learns the optimal combination from labeled data
— the industry-standard e-commerce ranking method. It subsumes our ad-hoc weights and can learn
context-dependent weighting (the leak-vs-honest tradeoff) that a single constant cannot express.

### The overfitting/leak hazard (design around it, do not ignore)
Training on the public set alone is dangerous: public is ~99% verbatim-leaked, so a model fit to it
will learn "coverage is king" — exactly what fails on the private/honest distribution, and the
organizer explicitly warns against overfitting public. **Mitigation — leak-balanced training:** we
can generate labeled sessions at any leak level (we own the generators + ground truth). Train on a
mix of public (leaky) + pillar_moderate (~21%) + pillar_free (~1%), so the model learns weights that
generalize across the leak spectrum, not the leak itself. Validate on a held-out leak-free split.

### Features (all already computable per candidate)
Per (session-final-state, candidate) — a small, leak-robust feature set (few features = less
overfitting):
- `retrieval_rank` — normalized position in the fused BM25+dense order (strong, leak-agnostic).
- `satisfaction` — max(lexical, semantic) constraint-match score.
- `coverage` — verbatim IDF coverage (the leaky signal; LTR should learn to trust it less alone).
- `cross_encoder` — ms-marco (query, product) precision score.
- `log_popularity`, `avg_rating` — priors.
- `price_proximity` — |price − budget| factor when a budget is disclosed.
- `category_match` — title matches the routed category.
- `n_constraints` / `specificity` — how much the shopper disclosed (context feature).

### Model
`sklearn` (lightgbm unavailable). **Pointwise logistic regression** over standardized features
(target=1, pool others=0, class-weighted), ranking by predicted P(relevant). Rationale: ~200–700
training sessions with distinct private users → a low-variance linear model generalizes far better
than gradient boosting and ships as a tiny weight vector (`cache/ltr_model.json`). Upgrade path:
pairwise (RankNet-style on feature diffs) or LambdaMART if `lightgbm` is later allowed.

### Implementation steps
1. `src/ranking_features.py` — `RankingFeatures.extract(asins, phrases, budget, retrieval_order,
   coverage, satisfaction, cross_encoder, query_text)` → `{asin: [features]}` + `FEATURE_NAMES`.
   Reuses the existing scorers so training and inference share one code path.
2. `scripts/collect_ltr_data.py` — replay public + pillar_moderate + pillar_free through the evaluator
   loop; at the final turn dump `(features, is_target, session_id, leak_level)` to `cache/ltr_data.jsonl`.
3. `scripts/train_ltr.py` — standardize, fit logistic regression, report per-feature weights and
   held-out (leak-free) ranking metrics; save `cache/ltr_model.json` (weights + means + stds).
4. Wire `USE_LTR` (default off): in `agent.respond`, after the base ranking, re-score the pool with
   the model and sort. Cross-encoder score is a feature, so LTR *replaces* the ad-hoc CE fusion.
5. **Decision gate:** adopt only if it beats the current default on pillar_moderate/free (leak-robust
   validation) while public stays ≥ ~0.89. Compare against hand-tuned baseline.

### Risks
Overfitting (mitigated by leak-balanced data + linear model + few features); label sparsity (one
positive per session — use class weights / consider pairwise); feature leakage (coverage is inherently
leaky — keep it one feature among many, let the model bound its weight).

---

## 2. Within-session relevance feedback

### Motivation
We accumulate constraints but do not use the *conversational* structure to refine retrieval. Standard
IR (Rocchio) and CRS: move the query representation toward confirmed/liked evidence and away from
rejected candidates each turn. Directly exploits the multi-turn signal we currently waste, and helps
exactly the browsing case (slow disclosure) without asking more questions.

### Design
- **Dense-vector Rocchio:** maintain a session query vector `q`. Each turn,
  `q ← α·q + β·centroid(confirmed constraint phrase embeddings) − γ·centroid(rejected-candidate
  embeddings)`, then re-retrieve with `q`. "Rejected" = items shown in prior turns that the customer
  moved past (the simulator's "not quite right" replies). Embeddings are already cached.
- **Constraint reinforcement:** attributes the shopper re-affirms across turns gain weight
  (ties into the ContextDistiller recency decay we already compute but underuse).

### Implementation steps
1. Track `state.shown_asins` (already partially via reveal) and `state.rejected_asins`.
2. `VectorRetriever.refine(q, positive_phrases, negative_asins, α, β, γ)` → new query vector;
   re-retrieve.
3. Flag `USE_RELEVANCE_FEEDBACK` (off); measure MTTC + hit on pillar_free browsing (the target case).

### Risks
Negative feedback is weak/noisy in the simulator ("not quite right" is generic). Keep γ small; guard
against drifting away from a target that was simply mis-ranked (relevance feedback assumes retrieval
recall — which we have at 99%, so this is safe).

---

## 3. Value-of-Information (VoI) clarification policy

### Motivation
The spec explicitly rewards "adaptive clarification and question-value estimation." Ours currently
asks `"other"` every turn (display mode) — effective for extraction but not a *policy*, and it never
decides ask-vs-recommend on expected value. A principled VoI policy is an expected, judged capability
we lack. (Note: our data shows extraction is not the bottleneck, so this is a comprehensiveness/
innovation play, not a big score lever — sequence it after LTR.)

### Design
Decision-theoretic ask-vs-recommend:
- Estimate `P(hit now)` from belief confidence (top-candidate margin / entropy over the pool).
- Estimate `E[hit | ask attribute a]` by simulating the pool split: for each askable attribute,
  how much would knowing its value concentrate the pool on the top candidate? (expected entropy
  reduction × its DECISION_WEIGHT).
- **Ask** iff `max_a E[gain(a)] − cost_of_a_turn > P(hit now)`; else **recommend**. The turn cost ties
  to MTTC/Efficiency, so the policy is directly optimizing the scored metric.
- Wire the chosen attribute to `ask_attribute` via `INFO_GAIN_MODE="ask"` (currently the selector is
  bypassed in display mode).

### Implementation steps
1. `src/clarification/voi.py` (or extend `QuestionSelector`): `should_ask(belief, pool) -> (bool, attr)`.
2. Expected-entropy-reduction estimate over the pool's attribute distributions (reuse
   `BeliefModel.attr_uncertainty` machinery).
3. Flag `USE_VOI_CLARIFY` (off); switch `INFO_GAIN_MODE` to `"ask"` under it; measure MTTC on
   pillar_moderate/free without regressing public MTTC.

### Risks
`ask` mode changes the `ask_attribute` contract behavior; must not increase MTTC on the leaky set
(where "other" extraction is already optimal). Gate carefully; keep "other" as the fallback.

---

## Sequencing & measurement

| Step | Build | Decision gate |
|---|---|---|
| 1 | LTR feature extractor + data collection | features reproduce/beat hand-tuned order offline |
| 2 | LTR train + wire | pillar_moderate/free up, public ≥ ~0.89 |
| 3 | Relevance feedback | browsing MTTC↓ / hit↑ on pillar_free, no public regress |
| 4 | VoI clarification | MTTC↓ on honest sets, public MTTC flat |

All measured via `scripts/eval_matrix.py` / `eval_default.py` across the leak spectrum. Each ships
behind its flag, off until it clears its gate. LTR is the foundation — its features are the shared
vocabulary the other two also benefit from.
