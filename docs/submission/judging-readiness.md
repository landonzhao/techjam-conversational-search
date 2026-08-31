# Judging Readiness Audit

Assessment of the repository at commit `d2ef469` after stabilization. This is an evidence review,
not a prediction of judge scores. Historical experiment figures are not treated as current unless
the final validation reproduced them.

## Technical Execution — 35%

### Concrete evidence

- The public contract is isolated in `starter/agent.py` and specified in
  `docs/agent_api_contract.json`.
- `src/agent.py` orchestrates distinct catalog, retrieval, state, ranking, belief, context, optional
  model, and tracing modules rather than placing every algorithm in the entry point.
- `src/catalog.py` builds a field-weighted SQLite FTS5 index; `src/retrieval.py` adds optional dense
  search and rank fusion; `src/ranking.py` contains explicit ranking stages and safeguards.
- `src/understanding.py` and `src/dialogue.py` preserve additions, rejections, corrections,
  no-preference boundaries, and overrides as session state.
- Optional dense, cross-encoder, Gemini, LTR, persistence, and tracing paths have deterministic
  fallbacks or are flag-gated.
- `scripts/eval_support.py` disables persistent DCP state and uses a temporary directory for
  independent benchmark runs.
- Automated evidence: 138 tests pass, Ruff lint passes, compilation passes, and the 200-session
  evaluation reproduces technical score `0.900068` with zero reported tokens.
- The trace UI exposes turn-level retrieval, ranking, belief, clarification, and reveal state for
  debugging and demonstration.

### Remaining weaknesses

- The orchestrator and two domain modules are large and contain many optional branches.
- Type checking is not a green gate; the exploratory MyPy run reports substantial pre-existing debt.
- Optional ML paths and caches were not exercised in the final deterministic validation.
- No formal latency, memory, concurrency, load, or browser/API test results exist.
- Phrase text, turn numbers, and new demotion weights are parallel lists; one optional invalidation
  path does not synchronize the weight list.
- Broad exception handling protects the scorer but can hide diagnostics outside tracing.

### Highest-value submission/presentation improvements

1. Demonstrate one complete live trace and show the contract, state revision, target rank, and
   zero-token fallback in under one minute.
2. Put the exact final validation table on one slide/Devpost section and explicitly distinguish
   passing gates from unresolved MyPy/Black debt.
3. Record cold-start time, warm-turn latency, and peak memory on the judging machine if time permits;
   report the commands and environment, not an estimate.
4. Rehearse the optional-model story carefully: implemented and graceful, but not part of the final
   deterministic score.

## Innovation and Problem Insight — 20%

### Concrete evidence

- `architecture.md`, `docs/COMPONENT_AUDIT.md`, `docs/DECISIONS.md`, and `docs/EXPERIMENTS.md` show
  sustained analysis of the mismatch between public simulator wording and natural shopper language.
- The repository includes `evaluator/robustness.py`, language/pillar stress datasets, oracle and
  ablation scripts, rather than optimizing only one headline public number.
- The system combines correction-aware state, intent routing, evidence-aware ranking, adaptive
  clarification/reveal, optional context persistence, and transparent tracing.
- The latest commit's soft override phrase demotion is a concrete response to the unusual evaluator
  semantics, while the ledger maintains a more general correction model.
- The system is deliberately model-optional instead of equating conversational AI with mandatory
  generative inference.

### Remaining weaknesses

- Several mechanisms are adapted from known retrieval/dialogue patterns; novelty lies in integration
  and measurement discipline, not a new learning algorithm.
- Some dialogue behavior, especially adaptive reveal and `ask_attribute="other"`, is shaped by the
  evaluator and may not represent ideal product UX.
- The robustness data is synthetic. The current config completed the full 250-session stress set,
  but the slower four-configuration full harness was only run on a 40-session subset.
- The repository contains many experimental flags, which can make the core innovation story diffuse.

### Highest-value submission/presentation improvements

1. Frame the insight as “explicit corrections plus graceful hybrid search,” not as a claim of a new
   model.
2. Use one override example to show why raw transcript accumulation fails and how active state fixes
   it.
3. State the simulator-overlap caveat before judges raise it; then show the separate robustness
   methodology as evidence of rigor.
4. Choose three differentiators for the presentation—correction-aware state, optional/offline model
   stack, and inspectable evaluation—and leave secondary flags in the appendix.

## Impact and Relevance — 20%

### Concrete evidence

- The product addresses a recognizable search failure: people often know desired outcomes but not
  catalog vocabulary and may revise needs during discovery.
- Buying, browsing, intent override, and no-preference boundary scenarios are explicit in
  `docs/competition_specification.md` and represented in tracked data.
- Explanations, clarifying questions, and trace visibility make results easier to inspect and
  correct than a one-shot ranked list.
- Offline fallback can improve accessibility in constrained-network or credential-free environments.
- The data package excludes raw identities, review text, timestamps, and product images according to
  `DATA_ATTRIBUTION.md` and `data/README.md`.

### Remaining weaknesses

- There is no user research, usability test, accessibility audit, or measured business/user outcome.
- The catalog is limited to one broad retail category and the evaluator seeks a known hidden target,
  which differs from open-ended real shopping.
- Persistent profile behavior has not been evaluated with returning real users.
- The UI is a developer trace tool rather than a shopper-ready experience.

### Highest-value submission/presentation improvements

1. Explain one concrete user journey—vague browsing to corrected preference to a relevant result—
   rather than claiming broad impact numerically.
2. Show how a user can correct the system and how the active state changes visibly.
3. Add only validated user/team feedback; avoid invented conversion, satisfaction, or reach metrics.
4. Mention the current category boundary and describe broader applicability as a hypothesis, not a
   completed result.

## Feasibility and Practicality — 15%

### Concrete evidence

- The baseline requires only Python 3.10+, SQLite FTS5, and a local catalog.
- The verified run makes no external API call and reports zero tokens.
- Heavy dependencies, model downloads, embeddings, and LTR artifacts are optional and ignored from
  Git; missing resources fall back to BM25/order preservation.
- Candidate pool/rerank depths are bounded, and expensive objects are initialized once per agent.
- Requirements are separated into deterministic, development, and optional groups.
- README includes setup, catalog validation, evaluator, CLI, UI, troubleshooting, and exact
  validation commands.
- Data/model/API limitations and the absence of production deployment are stated explicitly.

### Remaining weaknesses

- Catalog release URL/checksum publication is still a team placeholder; without it a fresh reviewer
  cannot complete setup from the tracked repository alone.
- Optional dependency ranges and model downloads are declared but not locked to a fully verified ML
  environment.
- There is no container, service deployment, index persistence, session eviction, authentication,
  telemetry, or privacy retention policy.
- Local JSON profile persistence is not suitable for concurrent production writes.

### Highest-value submission/presentation improvements

1. Publish the catalog archive and SHA-256 file, then perform the README on a clean machine.
2. Record the exact judging environment and whether optional models/caches are intentionally absent.
3. Keep the live demo on the deterministic path unless the optional environment has been rehearsed
   and measured.
4. Present a short productionization plan with explicit missing controls; do not imply that the
   local Flask tool is the deployment architecture.

## Presentation and Communication — 10%

### Concrete evidence

- Root README and `architecture.md` provide reviewer-oriented setup, diagrams, component maps,
  limitations, and reproduced results.
- `docs/submission/devpost-description.md` supplies a complete public-description draft with
  confirmed team credits and placeholders for unverifiable publication links.
- `docs/submission/demo-video-script.md` provides a timed narrative, verified demo path, recording
  checks, rights reminders, and YouTube visibility checks.
- `docs/submission/submission-checklist.md` separates complete work from team/publishing actions.
- The trace UI offers a visually understandable product story without requiring judges to read raw
  logs.

### Remaining weaknesses

- Repository URL, public video URL, catalog release URL, and team consent to public credits remain
  unresolved. The project title, contribution breakdown, and MIT source-code license are recorded.
- Older experiment/roadmap documents contain historical metrics and can confuse readers who skip
  their new historical-baseline warnings.
- The developer UI needs an accessibility/browser review and should not be presented as a polished
  production storefront.
- A dense flag ledger and many research scripts may distract from the core narrative.

### Highest-value submission/presentation improvements

1. Resolve every bracketed placeholder and perform a signed-out link check.
2. Use the same five numbers everywhere: sample count, Hit@10, MRR, MTTC, and technical score, plus
   zero tokens and the simulator-overlap caveat.
3. Keep the video focused on problem → live flow → architecture → evidence → limitations.
4. Put historical experiments in an appendix and direct first-time reviewers to README,
   `architecture.md`, and the trace UI.

## Overall readiness

The repository is technically demonstrable and substantially better prepared for review: the core
path runs, automated tests and lint pass, current metrics are reproduced, and submission prose is
complete. The largest remaining submission risk is **reproducibility outside the current workspace**
because the public catalog release URL/checksum and clean-machine verification are still missing.
The largest presentation risk is overstating public metrics despite the documented simulator-wording
overlap. Resolve those items before polishing secondary claims.
