# Submission Checklist

Status recorded against commit `d2ef469` on 31 August 2026. Items marked **TEAM INPUT** cannot be
verified from the repository and must be completed before submission.

## Repository and access

- [ ] **TEAM INPUT:** Confirm the repository is public and accessible while signed out.
- [ ] **TEAM INPUT:** Add the final repository URL: `[REPOSITORY_URL]`.
- [ ] Confirm the default branch points to the intended submission commit.
- [x] No commit, push, publication, upload, or Devpost submission was performed during cleanup.
- [x] Generated `data/catalog.jsonl`, caches, traces, results, `.env`, and Python caches are ignored.
- [ ] Decide whether `data/catalog.jsonl.gz` is a release asset or repository file; it is currently a
  pre-existing untracked file and was not changed by cleanup.
- [ ] Decide whether the pre-existing untracked `EXPERIMENT_LOG.md` and
  `scripts/check_regression.py` belong in the public submission.

## README and architecture

- [x] README contains purpose, solution, features, stack, setup, environment, commands, example flow,
  troubleshooting, limitations, future work, team contributions, and license status.
- [x] Root `architecture.md` documents actual components, data flow, APIs/models, configuration,
  fallbacks, security/privacy, performance, runtime, limitations, and future work.
- [x] Architecture includes Mermaid context and sequence diagrams.
- [x] Stale `docs/ARCHITECTURE.md` was moved to the required root location rather than duplicated.
- [x] Internal links were updated to the root architecture document.
- [ ] **TEAM INPUT:** Replace all bracketed URL placeholders.

## Installation reproducibility

- [x] Baseline Python requirement is documented as 3.10+.
- [x] `requirements.txt` declares the standard-library-only deterministic path.
- [x] `requirements-dev.txt` pins the locally verified test/lint/UI tool versions.
- [x] `requirements-optional.txt` declares optional ML and Gemini dependencies.
- [x] Catalog preparation, row-count verification, and checksum commands are documented.
- [ ] **TEAM INPUT:** Publish the catalog release asset and its `SHA256SUMS` file.
- [ ] **TEAM INPUT:** Add the public catalog release URL: `[CATALOG_RELEASE_URL]`.
- [x] Development dependencies and documented validation commands were successfully verified by the
  team in a fresh environment.
- [ ] Verify installation and imports for `requirements-optional.txt` in a separate fresh
  environment; live Gemini use requires a valid credential and is not a baseline requirement.

## End-to-end demo reliability

- [x] Representative two-turn evaluator workflow reproduced locally.
- [x] Verified target `B09PYB7B6Z` appeared at rank 1 on turn 2 for the first public sample.
- [x] Full 200-session public evaluation completed with zero reported model tokens.
- [x] Local Flask application imports and startup path are documented.
- [ ] Rehearse the exact video sample on the recording machine immediately before recording.
- [ ] Keep a terminal-only `scripts/chat.py` fallback ready.

## Submission prose and video

- [x] `devpost-description.md` draft completed.
- [x] `demo-video-script.md` completed with scene timing, narration, checklist, rights reminder, and
  YouTube visibility checks.
- [x] `judging-readiness.md` completed against all five weighted categories.
- [x] Devpost title/tagline confirmed as **TokenMaxx Copilot** — “Offline-first conversational search that
  turns changing preferences into ranked product recommendations.”
- [ ] **TEAM INPUT:** Edit the prepared prose into the platform's current fields.
- [ ] **TEAM INPUT:** Record and upload the demo video.
- [ ] **TEAM INPUT:** Add public YouTube URL: `[PUBLIC_YOUTUBE_URL]`.
- [ ] **TEAM INPUT:** Add the video URL to Devpost and verify the embed while signed out.

## Dataset, assets, privacy, and licensing

- [x] Amazon Reviews 2023 attribution is present in `DATA_ATTRIBUTION.md` and linked from README.
- [x] Tracked public data excludes raw identifiers, review text, timestamps, product images, and the
  private evaluation set according to repository documentation.
- [x] The trace UI uses inline original assets and does not fetch external media.
- [x] Credential-pattern scan found no API-key/private-key-shaped values in tracked source/docs.
- [x] `.env` is ignored and `.env.example` contains placeholders only.
- [ ] Manually inspect final video frames and repository history for personal data or credentials.
- [x] The team's original source code is licensed under the MIT License in the root `LICENSE` file.
- [ ] Verify code, dataset, model, API, and any added video-asset licenses are compatible.

## Team contributions

- [x] Team members are listed as Landon Zhao, Valerie Lim, and Bryan Koh.
- [x] Concrete engineering, evaluation, documentation, and submission contributions are attributed
  in the README and Devpost draft.
- [ ] Confirm all listed members consent to public names/credits.

## Baseline validation before cleanup

| Command | Result | Pre-existing status |
|---|---|---|
| `python3 -m pytest tests/ -v` | Failed before collection: selected Python had no pytest module. | Environment/tooling issue. |
| `pytest tests/ -v` | 138 passed. | Passing baseline through Conda pytest. |
| `python3 -m compileall -q src starter evaluator app scripts tests` | Passed. | Passing baseline. |
| `ruff check src starter evaluator app scripts tests` | Failed with 41 findings. | Pre-existing source-quality debt. |
| `black --check src starter evaluator app scripts tests` | Failed; 55 files would be reformatted. | Pre-existing; no formatter configuration existed. |
| `mypy src starter evaluator app scripts tests` | Failed on missing optional stubs and duplicate module discovery before full analysis. | Pre-existing configuration/dependency issue. |
| `python3 scripts/measure.py` | Passed: score `0.900068`. | README/old architecture had a stale higher score. |
| `python3 -m evaluator.robustness` | Interrupted after more than 90 seconds with buffered/no output. | Incomplete; no result claimed. |

## Final validation results

Update this table if any final command changes before submission.

| Command | Result |
|---|---|
| `ruff check src starter evaluator app scripts tests` | PASS. |
| `ruff check .` | PASS, including pre-existing untracked Python tooling. |
| `pytest tests/ -q` | PASS — 138 passed. |
| `python3 -m compileall -q src starter evaluator app scripts tests` | PASS. |
| `git diff --check` | PASS. |
| `python3 scripts/measure.py` | PASS — 200 sessions; Hit@10 `0.965`; MRR `0.852228`; MTTC `2.905`; efficiency `0.8095`; technical score `0.900068`; token usage `0`. |
| `python3 -u scripts/eval_default.py` | PASS — current flags reported without stale-attribute errors; public score `0.9001`; full 250-session language-stress score `0.7148`. |
| Representative two-turn workflow | PASS — first public target at rank 1 on turn 2. |
| Flask test-client smoke test | PASS — `/`, `/api/datasets`, `/api/samples`, and `/api/simulate` returned HTTP 200; selected sample produced two traced turns. |
| `/opt/anaconda3/bin/python app/trace_server.py --host 127.0.0.1 --port 5059` + localhost requests | PASS outside the managed network sandbox — server bound successfully; `/` and `/api/datasets` responded; process was stopped after the smoke test. |
| `python3 -m evaluator.robustness --limit 40` | PASS — all four configurations completed; each reported Hit@10 `0.725`, MRR `0.515`, MTTC `5.55`, score `0.6260` on this bounded no-optional-model run. |
| Fresh temporary virtual environment + `pip install -r requirements.txt` + FTS5 probe | PASS; the core manifest installs without third-party packages and FTS5 is available. |
| Fresh development environment + `requirements.txt` and `requirements-dev.txt` installation + documented development checks | PASS — completed successfully and reported by the team. |
| Current `eval_matrix` configuration/popularity helper smoke | PASS — obsolete dual-track attributes removed; current coverage/satisfaction rows configure without error. |
| `black --check ...` | Expected unresolved failure; large historical formatting diff, not made a gate. |
| `mypy --explicit-package-bases --ignore-missing-imports ...` | Expected unresolved failure; 92 pre-existing/dynamic typing errors in 21 files after removal of stale evaluation-script attributes. |

## Final manual review

- [x] Review `git status`, `git diff --stat`, and the complete diff.
- [x] Confirm the four recent ranking files still match commit `d2ef469` except mechanical cleanup.
- [x] Confirm no tests, lint rules, or type-check settings were weakened.
- [x] Confirm no generated, vendored, private, or large local artifact was unintentionally added.
- [x] Confirm README development commands in a fresh environment; completed successfully by the
  team.
- [x] Confirm every local Markdown link resolves.
- [x] Search for unresolved placeholders: `rg -n '\[[A-Z_]+\]|TEAM INPUT' README.md docs/submission`; remaining items are intentionally listed above for team input.
- [x] Repeat secret scan for the final cleanup diff; repeat once more immediately before publishing.
- [x] Confirm claims and metrics match the final run output.
- [x] Confirm application logic, output contract, and primary behavior are unchanged.
