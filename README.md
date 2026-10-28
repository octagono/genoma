# Genoma — Rare-Disease Variant-Prioritization Pipeline

**MVA Hackathon 2026** (Rare Disease, Real Kid) — Track 1 (variant prediction) +
Track 2 (drug repurposing) DSPy pipeline. **dspy==3.3.1** (pinned).

A deterministic-first, LLM-ranked framework for single-proband rare-disease
diagnosis — disease-agnostic architecture (adapting to another unsolved case
is a config change: known-gene set, HPO synonyms, mechanism description).

## Setup

```bash
uv sync                      # install from uv.lock (dspy 3.3.1 + extras)
cp .env.example .env         # add API keys
./run.sh demo                # offline smoke test (no keys needed)
```

On NixOS, use `./run.sh` (prepends the gcc `libstdc++.so.6` path so
litellm/tokenizers can load). Plain `uv run genoma ...` works elsewhere
(`uv run genoma ...` remains as a compatibility alias).

## Commands

| Command | Purpose |
|---|---|
| `./run.sh demo` | Offline end-to-end Track 1 run on synthetic data (DummyLM, BAMLAdapter) |
| `./run.sh ingest` | Download gated dataset (needs `HF_TOKEN`; free-space preflight vs real HF file sizes) |
| `./run.sh hpo` | Extract HPO terms from the phenotype docx (paragraphs + tables, negation-aware + semantic synonym map) |
| `./run.sh score` | Score candidates (Predict, parallel) + compound-het pairs → submission CSV |
| `./run.sh panel` | Heterogeneous multi-model consensus panel → submission CSV |
| `./run.sh candidates` | List/filter the deterministic candidate pool (read-only; gene/AF filters) |
| `./run.sh annotate` | Enrich candidates via Ensembl VEP REST (key-free; fails fast on network error) |
| `./run.sh phen2gene` | Rank the 6 MVA genes by phenotype fit (Phen2Gene REST API, fail-fast) |
| `./run.sh exomiser` | Exomiser comp-het analysis (external Java tool; `--yaml-only` to get the config) |
| `./run.sh phase` | Trans/cis phasing for compound-het pairs (VCF tags, trio Mendelian when family data present, or `--pipeline` for WhatsHap) |
| `./run.sh validate` | Score a submission CSV against the local answer key (rank points + F-max) |
| `./run.sh planner` | Submission strategy planner: record leaderboard feedback, re-weight, suggest pairs |
| `./run.sh optimize` | GEPA-optimize the scorer (synthetic trainset + LLM judge; `--judge-cls-weight` to rebalance) |
| `./run.sh evaluate` | Score a program on the devset with the judge (pre-submission check; same `--judge-cls-weight` knob) |
| `./run.sh track2` | ReActV2 drug-repurposing agent → `submission/track2_report.md` + methods report |
| `./run.sh compliance` | Print the data-deletion checklist |
| `./run.sh purge-cache` | Purge DSPy caches + gated dataset dir + HF hub cache entry |
| `./run.sh cost` | Dry-run budget estimate for every command (no API calls) |
| `./run.sh report` | Generate the Track 1 methods report (Markdown) for a submission |

Global flag: `--cost` reports token usage after any command.

Common scoring flags:
- `--strategy mixed|pairs|singles` — row mix (default mixed: compound-het pairs first, then singles)
- `--annotate` / `--no-annotate` — VEP-annotate candidates before scoring (default on)
- `--validate` — score the produced CSV against the synthetic answer key
- `--self-consistency N` — sample top candidates N× at temp 1.0, use the confidence-weighted mean EPCR (default 0 = off; each sample costs an LM call)
- `--max-pairs N` — number of compound-het pairs to score
- `--checkpoint PATH` — JSONL checkpoint (resume an interrupted run from the last chunk)
- `--resilient` — production mode: per-candidate failures degrade to a LOUD warning + conservative safe-default (epcr 0.0 / abstained) instead of aborting; enables checkpointing. Fail-fast remains the default.

## Architecture (source-verified against dspy 3.3.1)

```
config.py         all knobs: paths, model IDs, panel weights, submission limits, dates, priors
lm.py             LM factory (Anthropic prompt-caching), BAMLAdapter/native-FC defaults
cli.py            entrypoint: 18 commands (see above), --cost token reporting
cost.py           dry-run budget estimator (per-command call/token/cost plan)
net.py            shared transient-HTTP retry (429/5xx backoff; still fail-fast)
data/ingest.py    gated HF download (VCF + phenotype docx), HPO extraction (paragraphs + tables), VEP-CSQ header capture
data/annotate.py  deterministic VCF pre-filter (AF/depth/GQ/AR-homozygote) + VEP REST (REVEL/AlphaMissense/SpliceAI/CADD) + gene prior + mechanism KB + VEP-CSQ named-field parser
data/annot_cache.py local SQLite cache for VEP/Phen2Gene annotations (static coords; zero re-fetch cost)
data/phasing.py   trans/cis from VCF PS/HP/GT tags + trio Mendelian phasing (family VCFs) + WhatsHap pipeline generator
data/phen2gene.py Phen2Gene HPO->gene ranking (REST API; fail-fast, no silent fallback; SQLite-cached)
data/exomiser.py  Exomiser AUTOSOMAL_RECESSIVE_COMP_HET (phenopacket YAML + genes.tsv/variants.tsv parse + score-coupling merge)
data/submission.py Track 1 CSV writer/validator + pair/single row builders + strategy
data/demo.py      synthetic candidates + HPO + causal answer key for offline runs
checkpoint.py     chunked parallel scoring + JSONL resume + opt-in resilient mode (loud safe-default)
track1/scoring.py Predict scorer (BAMLAdapter, pydantic EPCR) + RLM scorer w/ sandbox domain tools + compound-het pairs
track1/metrics.py local rank points + F-max scoring (CAGI6 RGP rules) — validate before submitting
track1/consensus.py heterogeneous panel: deepcopy+set_lm, weighted median/trimmed-mean (PANEL weights), debate, spread-shrink, abstention
track1/strategy.py submission planner: record live feedback, re-weight EPCR, suggest pairs
track1/calibrate.py self-consistency EPCR + gene prior + submission-shape EPCR calibration (top trans pair ~0.99)
optimize/gepa_optimize.py  GEPA + composite LLM-as-judge metric (label-free) + synthetic trainset
track2/agent.py    ReActV2 tool-using agent (native function calling)
track2/tools.py    PubMed / ClinVar / OpenFDA DrugBank lookup tools (explicit dspy.Tool wraps)
track2/report.py   Track 2 Markdown report writer (the expert-judged artifact)
compliance.py      cache purge + deletion checklist (30-day rule)
mcp_server.py      MCP server: 22 typed tools for agentic pipeline consumption
```

### Key design decisions (research-backed)

- **Models** — PRIMARY_LM=`anthropic/claude-sonnet-5` (always-on adaptive
  thinking; reasoning tokens count toward max_tokens), SUB_LM=`anthropic/claude-haiku-4-5-20251001`,
  REFLECTION_LM=`claude-sonnet-5`. Panel: 4 providers (claude-sonnet-5 2.0x,
  gpt-5.6-sol, deepseek-v4-pro, gemini-3.7-flash). Panel availability drops
  members when their API key is unset; empty panel falls back to primary.
- **Pre-filter never drops splice/low-CADD causal variants** — the CADD cut
  applies only to confident coding variants (`_cadd_applies`); splice-region,
  intron, UTR and unknown-consequence variants bypass it (CAGI6 ASNS
  deep-intronic cautionary tale). The candidate-pool cap is config-driven
  (`cfg.CANDIDATE_MAX_DEFAULT`), not hardcoded, so the real genome-wide run
  cannot truncate the causal pair out of the pool.
- **Anthropic prompt caching is wired, not hoped for** — `PRIMARY_LM_KWARGS`
  passes `cache_control_injection_points` (system message + last message)
  so litellm injects cache breakpoints; the adapter places the static
  signature/schema/instructions prefix ahead of per-variant fields, which
  is what the endpoint caches across the 500-candidate parallel run.
  Combined with the DSPy disk cache, unchanged repeat steps are near-free.
- **Client-side token-bucket rate limiting** — every LM is wrapped in
  `RateLimitedLM`: it estimates input tokens per request and delays host-side
  to stay under the per-minute budget (`LM_TPM` env/config; 0 = unlimited).
  Prevents provider 429s / Tier-limit blocks during 8-thread parallel scoring
  instead of relying on retry-after-the-fact.
- **RLM not POT/CodeAct** — the latter are deprecated (removed 3.5); RLM is the
  3.3 flagship for structured/computed data. ProgramOfThought/CodeAct are not used.
  RLM = **Recursive Language Model**: the LLM explores large contexts through a
  sandboxed Python REPL (code execution + recursive sub-LLM calls) instead of
  stuffing everything in the prompt.
- **RLM runs with sandbox domain tools** — `RLM(tools=[mechanism_lookup,
  known_mva_gene, gene_prior_fraction])` injects deterministic MVA-domain
  lookups into the sandboxed interpreter (host-side, offline-safe, and they
  don't consume the `max_llm_calls` budget — only `llm_query` does).
- **Panel consensus not `dspy.Ensemble`** — the module doesn't exist in 3.3.1;
  `dspy.majority` is wrong for float EPCR. Use the custom panel (weighted
  median/trimmed-mean via the declared `PANEL` weights, debate on high
  spread, abstention when the panel can't converge post-debate).
- **GEPA needs a 5-arg metric returning `Prediction(score, feedback)`** — the
  judge in `optimize/gepa_optimize.py` follows that contract so the feedback
  string feeds GEPA reflection.
- **BAMLAdapter** (not plain JSONAdapter) enforces the pydantic `epcr` schema
  (ge=0, le=1) with ~2.4x smaller schema prompts on list-typed outputs.
- **Disk cache is hardened** — `configure_defaults()` calls
  `configure_cache(restrict_pickle=True)` with the `dspy.core.types.*` safe
  types, confining deserialization to an allowlist (plus the library-trusted
  litellm/openai/numpy prefixes). Out-of-allowlist entries are deleted and
  degrade to a cache miss, never a crash.
- **No labels → synthetic trainset + LLM-judge** — the honest optimization path
  for a single-proband hidden-answer challenge. Synthetic labels are derived
  from observable variant features (gene + AF + consequence), so GEPA has
  learnable signal.
- **Composite judge metric** — GEPA's judge scores calibration
  (|pred−true|) + classification (binary causal call right) with tunable
  weights, and only calls the LLM for feedback on imperfect scores (saves
  cost; GEPA reflects on the feedback string).
- **Panel availability** — OpenRouter members are dropped when
  `OPENROUTER_API_KEY` is unset; an empty panel falls back to the primary
  Claude model. The debate (chair) model resolves to the strongest available
  model: primary Claude if keyed, else the strongest panel member.
- **Candidates are pydantic** (`VariantCandidate`) so DSPy's typed-I/O (RLM,
  the adapters) serializes them correctly.
- **RLM execution is tested** — `tests/test_rlm.py` drives the full loop
  (code-gen → sandbox → SUBMIT → typed parse) offline with runnable code.

## MCP Server

The pipeline exposes all commands as typed MCP tools for agentic consumption.
Install the optional MCP dependency and run the server:

```bash
uv sync --extra mcp-server         # install mcp Python SDK
python -m genoma.mcp_server     # stdio mode (default)
```

22 tools registered (10 free/planning, 7 data-prep, 5 LM-powered with cost labels):

**Free tools** (no API cost): `demo`, `cost`, `compliance`, `validate`, `hpo`,
`planner`, `purge_cache`, `evaluate`, `candidates`, `mechanism`.

**Data preparation** (no LM cost): `ingest`, `annotate`, `phenopacket`,
`phen2gene`, `exomiser`, `phase`, `report`.

**LM-powered** (real API budget): `score` (~$2-3), `panel` (~$2-9 depending on
panel size), `rlm` (~$2-3), `optimize` (~$0.85-1.30), `track2` (~$0.10-0.15).

Cost labels in tool descriptions match `./run.sh cost` output.

## Scoring strategy (verified against official evaluation.py)

The CAGI6 RGP scorer (single-proband compression) rewards a *strictly unique*
rank-1 compound-het pair and penalizes ties:

- **Rank points** are a step function: `(1,100),(3,50),(5,25),(10,10)`. The
  cliff is rank 1 (100) → rank 2 (50).
- **F-max = 4/(4+W)** where W = distinct wrong variants ranked *above* the
  true pair. Rows ranked *below* are free, so fill the 10-row budget in
  confidence order.
- **EPCR ties at the top collapse F-max 1.0→0.667** — `calibrate.py` reserves
  a strictly-unique max (1.0) for the top pair and emits no ties anywhere.
- **Never split a compound-het into two single rows** (caps at 50), and never
  add a third variant to the pair row (100→50 AND F-max 1.0→0.8).
- **GEPA judge is classification-weighted (0.7/0.3)** — rank points reward the
  binary causal call, not exact EPCR calibration.

## Quality gates

- **Fail-fast policy** — no blind `except Exception` / `except: pass` in
  `src/`. Dependencies are declared in `pyproject.toml` (dspy, pydantic,
  cyvcf2, python-docx, python-dotenv, huggingface-hub) and imported at the
  top of modules. External calls (VEP REST, Phen2Gene API, PubMed/OpenFDA/
  ClinVar tools, Exomiser) propagate errors loudly instead of degrading to
  silent placeholders. `run.sh` must be run through `uv sync` after any
  pyproject change.
- **Lint + format** — `ruff check .` clean and `ruff format --check
  src tests` formatted (ruff is a declared dev
  dependency; config under `[tool.ruff]` in pyproject.toml).
- **Tests** — `uv run pytest tests/ -q` (134 offline tests) before every
  submission.
- **Automated provenance** — the submission CSV is produced end-to-end by
  the pipeline (pre-filter → VEP enrichment → scoring → calibration). No
  answer is injected or post-hoc curated; the ranked rows are the pipeline's
  own output, reproducible from the documented runbook.
- **Transient-HTTP retry, still fail-fast** — all external REST calls
  (VEP, Phen2Gene, PubMed, OpenFDA, ClinVar) go through
  `net.urlopen_with_retry`: 429/5xx and network blips back off and retry;
  4xx raises immediately and exhausted retries re-raise. No silent
  fallbacks.

## Cost

Run `./run.sh cost` for a per-command dry-run budget plan (no API calls made).
It models the pipeline's actual call structure — Predict per candidate, CoT
per compound-het pair, 4-member panel, GEPA metric evals, ReActV2 iterations
— against current list prices (verified 2026-08-30) and prints a low/high
spend bracket per command plus a full-hackathon total.

Order-of-magnitude spend (500 candidates, prices 2026-08-30):

| Command | Est. cost |
|---|---|
| `score` (500 candidates + 10 pairs) | ~$2–3 |
| `panel` (500 candidates; single model) | ~$2–3 |
| `panel` (500 candidates; 4 members, needs OPENROUTER key) | ~$8–12 |
| `optimize` (GEPA light) | ~$0.85–1.3 |
| `evaluate` (12-example devset) | ~$0.05–0.07 |
| `track2` (10 iters) | ~$0.10–0.14 |
| **Full hackathon** (3× score + 2× panel + optimize + track2, at the real `CANDIDATE_MAX_DEFAULT=2000` / `COMPHET_MAX_PAIRS=50` defaults) | **~$43–64** |

The biggest lever is `--max-candidates` — each 100 candidates ≈
$0.40–0.60, and the real genome-wide run uses `CANDIDATE_MAX_DEFAULT=2000`
(not the 500 in the per-command rows above), which is why the full-hackathon
figure is higher. Override prices with `MODEL_COST_IN` / `MODEL_COST_OUT` env
vars (per MTok) if a negotiated rate applies. The DSPy disk cache + Anthropic
prompt caching make repeat runs of unchanged steps near-free.

**Pricing caveats (verified 2026-08-30):** Sonnet 5 $2/$10 per MTok;
Haiku 4.5 $1/$5. `gpt-5.6-sol` is listed $2/$10 on OpenRouter (flex tier) but
$4/$20 standard on OpenAI — assume the higher figure unless you confirm the
tier. DeepSeek v4-pro uses time-of-day pricing ($1.32/$3.96 peak, half
off-peak) — the estimator uses peak (conservative). gemini-3.7-flash
$0.75/$3.75 through 2026-12-31, then $1.50/$7.50. OpenRouter adds no
per-request margin (only a one-time ~5.5% credit-top-up fee).

## Local VEP annotation

The challenge VCF is raw GATK (no gene/consequence). `vep_data/vep_run.sh`
annotates it with the official Ensembl VEP container (podman, release 110,
GRCh38 cache ~20 GB) → `data/annotated.vcf.gz`. The pipeline consumes the
annotated VCF automatically (`vcf_path()` prefers it); gene/consequence are
extracted from VEP's CSQ field by NAME (header-declared order — release-safe).
Deep predictors (gnomAD AF, REVEL, AlphaMissense, SpliceAI, CADD) stay on the
free VEP REST layer over the surviving candidate pool.

## Live runbook

`docs/LIVE_RUNBOOK.md` — the exact turnkey execution sequence for the real
dataset (ingest → HPO review → candidate pool → score/panel → the
6-submission leaderboard feedback-oracle loop → submit → purge), with
checkpoints, budget (~$43–64 full run via `./run.sh cost`), and the
hill-climb rules for exploiting full/partial-match feedback.

## Compliance

Submissions close **2026-10-24 23:59 UTC**; all data must be deleted within
**30 days of close** (email confirmation to the address in `config.py`).
`./run.sh purge-cache` clears all three genome-carrying caches: the DSPy
disk cache (`~/.dspy_cache`, stores LM responses verbatim), the gated
dataset dir (`data/`, including the `data/.cache/huggingface` download
blobs `hf_hub_download` leaves behind), and this dataset's entry in the
HF hub cache. `--keep-data` skips the dataset; `--keep-disk` skips the
DSPy cache. Relocate the DSPy cache via `DSPY_CACHEDIR` in `.env`.

## License

Code and documentation are released under **CC BY 4.0** (see `LICENSE`,
full legal code, and `NOTICE` for the required acknowledgement), matching the
hackathon rule that submissions are released under a CC-BY license. This
repository contains **no patient data**: the gated dataset and every
genotype-scale derivative are gitignored and destroyed under the 30-day
deletion rule.
