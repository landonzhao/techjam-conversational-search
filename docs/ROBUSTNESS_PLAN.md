# Robustness Plan — replace brittle hand-built components with graceful degradation

Goal: reduce dependence on enumerated hand-written lists that silently fail on unseen
language and products. The principle throughout: **offline LLM for understanding, cached,
with a deterministic fallback** — so the system handles novel input instead of only the
inputs someone remembered to type into a table.

This is a *general robustness* effort, not test-set tuning. Several items here will NOT move
the current leaked evaluator (it discloses verbatim text and we drain constraints with
`ask_attribute="other"`), so they are measured on a **language-stress held-out set** (§5),
not on the public score. That distinction is the whole point.

---

## 1. Inventory of brittle hand-built components

Measured sizes in the current code:

| Component | Where | Size | Failure mode on unseen input |
|---|---|---|---|
| `EXPANSIONS` (synonyms) | understanding.py | **16** | "merino", "bamboo", "athleisure", "bodycon" → no expansion |
| `USE_CASE_LEXICON` | understanding.py | **19** | "recital", "communion", "gymnastics", "baby shower" → no implied attributes |
| `USE_CASE_KEYS` (detection) | understanding.py | **17** | occasion not in list → never detected as a use_case at all |
| `MATERIAL_RE` / `COLOR_RE` / `STYLE_RE` | understanding.py | 20 / 21 / 18 | "chambray", "ecru", "peplum" → not extracted |
| `CATEGORY_CANON` | understanding.py | 83 → 31 | unlisted head noun → category unknown → wrong required-slots |
| `REQUIRED_SLOTS` (which slots matter) | understanding.py | **16 categories** | 17th+ category → generic `DEFAULT_REQ` for everything |
| `DECISION_WEIGHT` + `ASK_PRIORITY` | understanding.py / config | 7 slots, fixed | **category-blind**: a wedding dress and running socks get identical slot priority |

Graceful-degradation paths that ALREADY exist (keep, extend):
- `LLMSlotExtractor` — LLM slot parse, fires when regex finds <2 slots. Covers the
  MATERIAL/COLOR/STYLE regex gap. Conservative gating.
- `SmartUseCaseInferencer` / `LLMUseCaseInferrer` — wraps `USE_CASE_LEXICON`, LLM extends it.
- `LLMResponseGenerator`, listwise `LLMReranker`.
- `cache/synonyms.json` via `build_synonyms.py` — data-driven expansion (embedding neighbours).

So the pattern is established. The gaps are the components with **no** graceful fallback:
`REQUIRED_SLOTS`/`DECISION_WEIGHT`/`ASK_PRIORITY` (category-blind questioning) and the
occasion **detection** (`USE_CASE_KEYS`), which fails silently before the LLM inferrer runs.

---

## 2. The recurring weakness (root cause)

Every brittle component is an **enumeration**: it works for members of a list and returns
nothing for non-members, silently. Enumerations do not degrade — they cliff. The robust
alternative is a **function that always returns a reasonable answer**: a data-driven or
LLM-backed component that maps *any* input to an estimate, with a static table as the fast
path and a default as the floor. Three tiers, always:

```
known input      -> hand table / cache   (fast, free, exact)
unseen input     -> LLM or data-driven    (cached; handles novelty)
LLM unavailable  -> sensible default       (never crashes, never empty)
```

---

## 3. Prioritised fixes

### R1 — Category-adaptive questioning (highest robustness value)
**Brittle now:** `REQUIRED_SLOTS` covers 16 categories; everything else falls to a single
`DEFAULT_REQ = [material, color, style, use_case]`. `DECISION_WEIGHT` is global. So for a
"kids snow boot" or "tennis bracelet" we ask about material/color like it's a generic top,
never the category-defining attribute (age/size for kids, stone/metal for jewellery, activity
for shoes).

**Graceful replacement:** derive the decision-defining slots per category from two robust,
distribution-independent sources, blended:
1. **Data-driven (no LLM):** for products in the resolved category, which attributes vary
   most across the catalog (highest entropy) are the ones worth asking about. Precomputed
   once from the frozen catalog. Works for *every* category that exists in the data.
2. **LLM prior (cached, offline):** for each canonical category bucket (~31 + any resolved
   at runtime), one cached call returns the ranked decision-defining slots ("wedding dress →
   occasion, silhouette, size"). Falls back to the data-driven set, then `DEFAULT_REQ`.

`QuestionSelector` reads the category-specific slot priority instead of the global
`DECISION_WEIGHT`. Fully backward-compatible: unknown category still yields a sensible set.

### R2 — Occasion detection that does not cliff
**Brittle now:** `USE_CASE_KEYS` (17 words) gates occasion detection; if the shopper's
occasion is not a literal member, `SmartUseCaseInferencer` never fires because no use_case
slot was created. The LLM inferrer is downstream of a hard-coded detector.

**Graceful replacement:** when the message has no recognised use_case but is not trivial, let
the LLM slot extractor (already built) emit a `use_case` value from free text, which then
feeds the LLM use-case inferrer. Detection becomes LLM-backed, not list-gated. Cache by
message.

### R3 — Widen the LLM slot-extraction trigger (small)
**Brittle now:** `LLMSlotExtractor` fires only when regex finds <2 slots. A message with 2
regex hits plus a paraphrased third constraint ("...and something that breathes well") skips
the LLM and loses the third. **Fix:** also fire when the message length/complexity suggests
un-extracted content, not only on low regex count. Measure token cost vs recall gain.

### R4 — Retire / data-drive the synonym table (lower priority)
**Brittle now:** 16 `EXPANSIONS`. BUT the synthetic diagnosis showed retrieval recall is
already 98.7%, so expansion is not currently a bottleneck. **Do not hand-write more entries.**
If touched at all: lean entirely on the existing data-driven `cache/synonyms.json`
(`build_synonyms.py`) and delete most of the hand list, so expansion is catalog-derived, not
memorised. Low priority — recall is not where we lose.

### R5 — Offline product facet enrichment (foundation, conditional)
The deepest robustness upgrade: one cached LLM pass over the catalog emitting structured
facets per product (silhouette, pattern, occasion, age-group, metal/stone, closure), schema
**per top-level category** (apparel / footwear / jewellery / kids — not one apparel-centric
schema). This feeds R1 (better decision slots), ranking, and diversity. Expensive (batch +
QA), so gated: build only after R1/R2 show that better *understanding* actually moves the
language-stress set. Not a blind 50k batch on faith.

---

## 4. What we explicitly will NOT do

- Add more hand-written synonym / use-case / category entries. That is fitting harder, the
  opposite of robustness.
- Replace the regex fast paths entirely. They are correct and free when they match; keep them
  as tier-1 and add the LLM/data tier beneath. Do not route every message through an LLM.
- Build R5 before R1/R2 prove understanding is the bottleneck on the stress set.

---

## 5. How we measure general robustness (not test fitting)

Enumerations pass the current evaluator because it discloses catalog-verbatim text. To see
the brittle failures we must **stress language specifically**:

- **Build a language-stress held-out set** (extends the paraphrase harness): occasions and
  attributes expressed in natural, non-catalog words ("my daughter's recital", "keeps me warm
  without the itch", "goes with a suit"), plus kids/jewellery/footwear categories under-served
  by the tables. This is the set R1–R3 are optimised against.
- **Track the gap, not the top-line.** Health metric = (public score − stress-set score). A
  shrinking gap is real generalisation; a high public score with a wide gap is overfitting.
- **Discipline unchanged:** every change behind a flag, measured on public + robustness +
  synthetic + the new stress set; kept only if it helps the held-out sets without regressing
  public. LLM paths always retain the deterministic fallback and count token cost.

---

## 6. Sequence

| Order | Item | Effort | Measured on | Robustness value |
|---|---|---|---|---|
| 1 | Language-stress held-out set (§5) | S | build first | enables everything |
| 2 | R2 occasion detection (LLM-backed, cached) | S | stress set | high |
| 3 | R1 category-adaptive questioning | M | stress set + synthetic | high |
| 4 | R3 widen slot-extraction trigger | S | stress + token cost | medium |
| 5 | R4 data-drive synonyms / delete hand list | S | recall guardrail | low |
| 6 | R5 per-category facet enrichment | L | stress + synthetic | conditional on 2–3 |

**Expected:** R1–R3 shrink the public↔stress gap by handling unseen language gracefully
rather than enumerating it. They may barely move the leaked public score — that is fine and
expected; the point is a system that stays intelligent on inputs no one hard-coded, which is
what a broad private set and real users will actually send.
