# Real-Data Live Runbook — MVA Hackathon 2026

Turnkey execution sequence for the moment HF token + API keys are available.
Submissions close **2026-10-24 23:59 UTC**; Track 1 limit **6 per person**
(Track 2: 3 per team — raised from 1 by organizer update 2026-08-29; only the
LATEST Track 2 submission is judged). Use the leaderboard feedback oracle
between submissions.

## Prereqs

- `.env`: `HF_TOKEN`, `ANTHROPIC_API_KEY`, optional `OPENROUTER_API_KEY`
  (4-provider panel; **omit it for the minimal ~$3 run**), optional `LM_TPM`.
- Storage: preflight runs automatically (`ingest` checks HF file sizes +20%
  headroom vs free disk). FastQ downloads need ~100 GB; VCF+docx only need
  ~5 GB.

## Minimal submission path (~$3, the "get into the perfect-score club" run)

Once the VEP cache has finished downloading, one command produces a
competitive submission:

```bash
bash scripts/run_minimal_submission.sh
```

This moves + extracts the cache, VEP-annotates the real VCF, scores ~500
candidates single-model (no panel), calibrates, and writes
`submissions/minimal_submission.csv`. Budget ~$3-5. The BUB1B compound-het
pair should surface in the top ranks from VEP + gnomAD + deep predictors
alone — the panel is a precision luxury, not a requirement for a valid
100.0/1.000 submission.

## Phase 1 — Ingest + phenotype

```bash
./run.sh ingest          # VCF + .tbi + phenotype docx (preflight + fail-fast)
./run.sh hpo             # negation-aware + semantic HPO extraction; REVIEW output
```

**Checkpoint:** manually review the extracted HPO list against the clinical
docx. Wrong/negated phenotypes poison Phen2Gene + scoring.

## Phase 2 — Local annotation + candidate pool (the rank-points win condition)

The challenge VCF is raw GATK (no gene/consequence). Annotate it once with
VEP (podman, official image):

```bash
nix-shell                              # tooling env (bcftools/tabix)
bash vep_data/vep_cache_download.sh    # one-time ~20 GB GRCh38 cache
bash vep_data/vep_run.sh               # annotate 5M variants -> data/annotated.vcf.gz
```

Then build + enrich the candidate pool (`vcf_path()` auto-uses the annotated
VCF when present):

```bash
./run.sh candidates     # read-only pool inspection (gene/AF/consequence)
./run.sh annotate       # VEP REST: gnomAD AF + REVEL/AlphaMissense/SpliceAI/CADD (SQLite-cached)
```

**Checkpoint:** confirm BUB1B/CEP57/TRIP13/MAD1L1/MAD2L1BP/CEP192 have
plausible AR-in-trans pairs in the pool, AND that novel-gene pairs survive
(pre-filter is gene-agnostic + splice-low-CADD-safe). If the causal gene were
novel (SMC5 precedent), it is still in the pool.

## Phase 3 — Score + panel + calibrate

```bash
./run.sh score --annotate --self-consistency 5 --checkpoint submissions/score_ckpt.jsonl
# or: ./run.sh panel --checkpoint submissions/panel_ckpt.jsonl
./run.sh validate         # local rank points + F-max vs synthetic key (sanity only)
./run.sh cost             # confirm budget before the real run
```

**Production mode:** add `--resilient` for the final run (per-candidate
failures degrade loudly + continue; JSONL resume across interruptions). The
pre-filter + weighted panel + strict-unique-EPCR calibration guarantee the
top pair is rank 1 with no ties (ties collapse F-max 1.0→0.667).

## Phase 4 — The feedback-oracle loop (this is how the field separates)

The leaderboard returns per-submission: "Full match at rank X" / "Partial
match (one of two) at rank X" / "No match". With 6 submissions, hill-climb:

```bash
./run.sh score --out submissions/s1.csv      # submission 1: full ranked list
# leaderboard: e.g. "Partial match at rank 2"  -> one true variant IS in your list
./run.sh planner --record submissions/s1.csv --matched "chr2:47908123:G>A" \
    --note "partial at rank 2"
./run.sh score --strategy pairs --out submissions/s2.csv   # re-weighted, pair-focused
```

Rules of the loop:
- A **full match** at rank 1 → you're done; consider submitting the same with
  different EPCR ordering to confirm (or use remaining slots on Track 2).
- A **partial match** tells you exactly one of the two true variants is in
  your list → re-rank to promote that gene's pairs + broaden the same-gene
  candidate search (the second variant is in-trans in the same gene).
- **No match** → your pool is wrong (pre-filter dropped it, or novel gene
  beyond candidate cap). Loosen the pre-filter / raise `CANDIDATE_MAX_DEFAULT`
  / re-check HPO, not the model.
- Always keep the compound-het pair as ONE row (full rank points require it;
  never split into singles).

## Phase 5 — Submit + compliance

The submission form requires files named with the convention
`<username>_<model>.<ext>` (per the FAQ: e.g. `octagono_gatk-haplotype.csv`,
`octagono_track1_report.pdf/.md`). Run the generators, then rename to your
username + model tag:

```bash
./run.sh validate --out submissions/sN.csv    # final sanity
./run.sh report submissions/sN.csv --team "YOUR_TEAM" \
    --out "submissions/octagono_track1_report.md"   # Track 1 methods (11 Qs)
./run.sh track2 --team "YOUR_TEAM"            # Track 2 agent -> reports
# rename track2 outputs to octagono_track2_report.md / ..._methods_report.md
./run.sh compliance                            # 30-day deletion checklist
```

**Judge-visible deliverables (the prize is judged over 2-3 months — these
matter as much as the score):**
- `submissions/track1_report.md` — answers all 11 Track 1 template questions
- `submission/track2_report.md` + `submission/track2_methods_report.md` —
  the drug report + all 11 Track 2 template questions
- `docs/PITCH_SCRIPT.md` — 3-minute video script (the required pitch)
- Public GitHub repo (README documents the full pipeline)

After the final submission: `./run.sh purge-cache` clears DSPy cache + gated
dataset + HF hub cache + annotation cache (genome-carrying material).

## Budget notes (real API)

- Full run (2000 candidates + 50 pairs, panel + optimize + track2) ≈
  **$43–64** per `./run.sh cost` (claude-sonnet-5 primary). Run `./run.sh
  cost` for exact numbers before committing.
- The SQLite annotation cache + DSPy disk cache + Anthropic prompt caching
  make repeated runs near-free after the first.
- 6 submissions × ~$10-13 ≈ $60-80 total — well within the prize/credit
  envelope; `LM_TPM` token-bucket (config) prevents 429-tier surprises.
