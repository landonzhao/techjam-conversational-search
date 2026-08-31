# TokenMaxx Copilot Demo Video Script

## Recommended structure

Target **2:45–3:15**, subject to the submission platform's actual limit. Record a clean continuous
product flow first, then add only short architecture and results overlays. If the limit is three
minutes, shorten the architecture and next-steps scenes rather than speeding up narration.

## Scene-by-scene script

### 0:00–0:18 — Problem

**Visual:** Title card, then a simple shopper query in the trace UI or terminal.

**Narration:**

> Product search assumes shoppers know the catalog's keywords. Real shoppers start vague, describe
> an occasion, decline to choose an attribute, or change their minds. TokenMaxx Copilot turns that evolving
> conversation into an explicit need and ranked product recommendations.

### 0:18–0:35 — Solution

**Visual:** Show the trace UI dataset and scenario selectors. Briefly highlight the conversation and
inspector panes.

**Narration:**

> Our offline-first agent has up to ten turns to find one target product in a 50,000-item catalog. It
> combines retrieval, correction-aware preference state, ranking, clarification, and adaptive result
> reveal. It works without an API key, while optional local and hosted models can improve language
> handling when available.

### 0:35–1:32 — End-to-end product demonstration

**Visual:** Start the app with `python3 app/trace_server.py`. Select `public_set` and the first buying
sample, then click **Simulate**. Keep the browser zoom large enough to read the conversation.

**Narration:**

> The shopper begins: “I'm looking for Jewelry Necklaces. A key requirement is: Material: alloy.”
> The agent retrieves and ranks candidates, but confidence is still low, so it asks for another
> differentiating preference and reveals only one result. That prevents a weak early list from
> becoming the final answer.

**Visual:** Let turn two appear. Click the agent turn to show the inspector. Highlight the active
constraints, retrieval/ranking summary, and target rank.

**Narration:**

> The next answer reveals a distinctive feature. TokenMaxx Copilot updates its state, reruns retrieval and
> need-satisfaction ranking, and returns target `B09PYB7B6Z` at rank one. This verified flow used zero
> model tokens.

If the selected sample behaves differently after an intentional code/config change, record the live
result and update this script. Do not splice in a result the current checkout cannot reproduce.

### 1:32–1:58 — Intent revision and reliability

**Visual:** Switch to an intent-override sample, or show a prepared saved trace produced by the
current code. Highlight old/new preference state rather than waiting through all turns.

**Narration:**

> When a shopper changes direction, the agent records an explicit update instead of treating every
> earlier word as permanently valid. The current ranking path softly demotes older simulator phrases
> so the new requirement dominates while still retaining weak corroborating evidence. Independent
> benchmark sessions run without persistent profile state, preventing score contamination.

### 1:58–2:25 — Architecture

**Visual:** Show the Mermaid diagram from `architecture.md` or a clean exported screenshot of it.

**Narration:**

> Each turn follows a staged pipeline: intent and constraint updates, SQLite FTS5 BM25 retrieval,
> optional BGE dense search and rank fusion, need-satisfaction ranking, then belief, clarification,
> and reveal policy. Optional cross-encoder and Gemini components fail safely to the deterministic
> path, so missing models, network, or credentials do not stop the demo.

### 2:25–2:45 — Engineering evidence

**Visual:** Terminal showing the compact validation summary.

**Narration:**

> The repository has 138 passing tests and clean Ruff lint. On the frozen 200-session public set,
> the reproduced deterministic run reached 0.965 Hit Rate at ten, 0.852 MRR, 2.905 mean turns to
> conversion, and a 0.900 technical score with zero reported tokens.

Keep the caveat visible in small text: “Public simulator wording overlaps target metadata; this is
not a production-performance claim.”

### 2:45–3:05 — Impact, feasibility, and innovation

**Visual:** Return to the UI and briefly show the explainable top pick plus inspector.

**Narration:**

> The value is practical, inspectable conversational search for people who know the outcome they
> want but not the catalog vocabulary. The core is local and low-dependency; semantic and generative
> components are modular rather than mandatory. Our main differentiation is correction-aware state,
> evidence-based ranking, and transparent graceful degradation.

### 3:05–3:15 — Limitations and next steps

**Visual:** Final card with repository and demo links.

**Narration:**

> Next we would validate with human-written requests, measure cold and warm resource use, improve
> typing and modularity, and add production privacy, concurrency, and deployment controls.

If constrained to three minutes, cut this scene to one sentence.

## Alternative terminal-only demo

If the browser demo is unreliable, use:

```bash
python3 scripts/chat.py --top-k 5
```

Prepare two short shopper turns that exercise one category and one distinctive constraint. Do not
promise a particular ID until the rehearsal confirms it on the recording machine.

## Recording checklist

- [ ] Use the exact final commit and a clean terminal prompt.
- [ ] Confirm `data/catalog.jsonl` has 50,000 rows.
- [ ] Rehearse the selected sample after clearing only generated trace output, not model/data assets.
- [ ] Start the local server before recording and verify `http://127.0.0.1:5001`.
- [ ] Hide API keys, `.env`, shell history, personal paths, notifications, email, and unrelated tabs.
- [ ] Use readable browser zoom and terminal font size.
- [ ] Show at least one complete input → clarification → output flow.
- [ ] Show a real response, target rank, and zero-token usage from the current run.
- [ ] State that optional models are optional and that the final deterministic score did not use them.
- [ ] Include the public-simulator overlap caveat.
- [ ] Keep architecture explanation under 30 seconds.
- [ ] Add captions and check audio intelligibility.
- [ ] Verify links on the final card.
- [ ] Watch the exported video from beginning to end before upload.

## YouTube upload and visibility checklist

- [ ] Export in a common format such as MP4/H.264 at 1080p when practical.
- [ ] Upload to the correct team/channel account.
- [ ] Set visibility to **Public** or the exact visibility required by the competition; do not leave
  it Private.
- [ ] Add a concise title, project summary, repository link, dataset attribution, and team credits.
- [ ] Wait for HD processing to finish.
- [ ] Test the link in a signed-out/private browser window.
- [ ] Confirm captions, thumbnail, and description render correctly.
- [ ] Paste the final public link into Devpost and `submission-checklist.md`.
- [ ] Re-open the Devpost entry and test the embedded video.

## Rights and attribution reminder

Do not use unlicensed trademarks as endorsements, copyrighted music, stock footage, product images,
screenshots, fonts, datasets, or other assets. The project UI currently needs none of those. If the
video displays Amazon Reviews 2023-derived product text, include the attribution from
`DATA_ATTRIBUTION.md` and do not imply Amazon sponsorship. Obtain consent before showing team-member
faces, voices, names, or personal accounts.
