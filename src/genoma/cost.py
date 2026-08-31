"""Cost estimator — dry-run budget for each pipeline command.

Prints a per-command call/token/cost plan BEFORE spending any API budget.
The estimates are grounded in the pipeline's actual structure:

- `score`: 1 PredictVariantScorer call per candidate + 1 CoT call per
  compound-het pair (ChainOfThought adds `reasoning` tokens).
- `panel`: 1 call per (candidate, panel member) + a debate round on high
  spread.
- `optimize`: GEPA metric evals; each eval = 1 scorer call + (on imperfect
  score) 1 judge-feedback LLM call. `auto=light` caps metric calls.
- `track2`: ReActV2 agent, ~1-2 LLM calls per iteration (max_iters capped).
- `evaluate`: 1 call per devset example with the judge.
- `rlm`: sandboxed code-gen; call count is dynamic (max_llm_calls cap).

Model prices are overridable via env `MODEL_COST_IN=`/`MODEL_COST_OUT=`
(per MTok) when a different rate applies.
"""

from __future__ import annotations

import os

from genoma import config as cfg

# ── Per-model pricing ($ per 1M tokens) ────────────────────────────────────
# Keyed by the config model string; falls back to env overrides for the
# primary model so a different rate can be plugged in.
MODEL_COST: dict[str, tuple[float, float]] = {
    # anthropic/... (direct)
    "anthropic/claude-sonnet-5": (2.00, 10.00),
    "anthropic/claude-haiku-4-5-20251001": (1.00, 5.00),
    # openrouter/... slugs
    "openrouter/anthropic/claude-sonnet-5": (2.00, 10.00),
    # NOTE: $2/$10 is OpenRouter's flex-tier list price (OpenAI's standard
    # on-demand is $4/$20) — the estimate uses the lower (best-case) figure.
    "openrouter/openai/gpt-5.6-sol": (2.00, 10.00),
    # DeepSeek v4-pro peak price; off-peak is half — use the peak figure so
    # the estimate is conservative.
    "openrouter/deepseek/deepseek-v4-pro": (1.32, 3.96),
    "deepseek/deepseek-v4-pro": (1.32, 3.96),  # direct DeepSeek API, same prices
    "openrouter/google/gemini-3.7-flash": (0.75, 3.75),
}

# ── Token-size model (per call, in/out) — grounded in prompt structure ──────
# variant scoring prompt ~1.2-1.8k in (signature + variant evidence + HPO +
# gene mechanism + known-gene list); output ~150-250 (epcr + rationale).
TOKENS_SCORE_IN = 1500
TOKENS_SCORE_OUT = 200
# Pair scoring adds a second variant + mechanism cross-check; CoT adds reasoning.
TOKENS_PAIR_IN = 2500
TOKENS_PAIR_OUT = 400
# Panel: same scoring prompt per member; debate adds a round on high spread.
TOKENS_DEBATE_IN = 3000
TOKENS_DEBATE_OUT = 500
# GEPA judge feedback call (~2.5k in: example + prediction + trace).
TOKENS_JUDGE_IN = 2500
TOKENS_JUDGE_OUT = 300
# ReActV2: per-iteration prompt ~2k in (tool results accumulate), ~400 out.
TOKENS_AGENT_IN = 2000
TOKENS_AGENT_OUT = 400
# RLM: sandbox code-gen; per-iteration ~2.5k in, ~600 out, capped by max_llm_calls.
TOKENS_RLM_IN = 2500
TOKENS_RLM_OUT = 600


def _price(model: str) -> tuple[float, float]:
    """Return (cost_in_per_mtok, cost_out_per_mtok) for a model string."""
    if model in MODEL_COST:
        return MODEL_COST[model]
    # Fall back to env overrides (primary model), else a mid-range default.
    fin = os.environ.get("MODEL_COST_IN")
    fout = os.environ.get("MODEL_COST_OUT")
    if fin and fout:
        return (float(fin), float(fout))
    return (2.00, 10.00)


def _cost(model: str, tin: int, tout: int) -> float:
    pin, pout = _price(model)
    return (tin / 1_000_000) * pin + (tout / 1_000_000) * pout


def _fmt_usd(v: float) -> str:
    return f"${v:,.2f}" if v >= 0.01 else f"${v * 100:.1f}¢"


def _fmt_bracket(low: float, high: float) -> str:
    return f"{_fmt_usd(low)} – {_fmt_usd(high)}"


def _line(model: str, calls: int, tin: int, tout: int, low_scale=1.0, high_scale=1.0) -> dict:
    """One cost line: calls x tokens x price, with a low/high bracket."""
    low = _cost(model, int(tin * low_scale), int(tout * low_scale)) * calls
    high = _cost(model, int(tin * high_scale), int(tout * high_scale)) * calls
    return {
        "model": model,
        "calls": calls,
        "tokens_in": tin * calls,
        "tokens_out": tout * calls,
        "low": low,
        "high": high,
    }


def _fmt_line(line: dict) -> str:
    return (
        f"  {line['model']:<42s} {line['calls']:>5d} calls "
        f"{line['tokens_in']:>9,} in / {line['tokens_out']:>6,} out  "
        f"{_fmt_bracket(line['low'], line['high'])}"
    )


# ── Per-command plans ───────────────────────────────────────────────────────


def plan_score(
    n_candidates: int | None = None, n_pairs: int | None = None, rlm: bool = False
) -> list[dict]:
    """`score` — Predict per candidate + CoT per pair (+ optional RLM)."""
    n_candidates = n_candidates or cfg.CANDIDATE_MAX_DEFAULT
    n_pairs = n_pairs or cfg.COMPHET_MAX_PAIRS
    lines = [
        _line(cfg.PRIMARY_LM, n_candidates, TOKENS_SCORE_IN, TOKENS_SCORE_OUT, 0.8, 1.2),
        _line(cfg.PRIMARY_LM, n_pairs, TOKENS_PAIR_IN, TOKENS_PAIR_OUT, 0.8, 1.2),
    ]
    if rlm:
        # RLM is dynamic; budget the cap (RLMVariantScorer max_llm_calls=40)
        # at the low end and assume ~1/3 of the budget at the high end.
        lines.append(_line(cfg.PRIMARY_LM, 40, TOKENS_RLM_IN, TOKENS_RLM_OUT, 0.10, 0.35))
    return lines


def plan_panel(
    n_candidates: int | None = None, n_pairs: int | None = None, debate: bool = True
) -> list[dict]:
    """`panel` — each member scores each candidate + pair (+ debate on spread).

    Uses the availability-resolved panel (OpenRouter members dropped when
    their key is unset; empty -> single primary-model fallback) so the cost
    estimate matches what a real `panel` run would spend.
    """
    from genoma.track1.consensus import _resolve_panel

    n_candidates = n_candidates or cfg.CANDIDATE_MAX_DEFAULT
    n_pairs = n_pairs or cfg.COMPHET_MAX_PAIRS
    resolved = _resolve_panel(cfg.PANEL)
    lines = []
    for slug, _w in resolved.items():
        lines.append(_line(slug, n_candidates, TOKENS_SCORE_IN, TOKENS_SCORE_OUT, 0.8, 1.2))
        lines.append(_line(slug, n_pairs, TOKENS_PAIR_IN, TOKENS_PAIR_OUT, 0.8, 1.2))
    if debate and len(resolved) > 1:
        chair = cfg.PRIMARY_LM
        # Debate only fires when spread > threshold (~a minority of cases);
        # budget ~20% of candidates through a debate round.
        lines.append(
            _line(
                chair,
                max(1, int(n_candidates * 0.2)),
                TOKENS_DEBATE_IN,
                TOKENS_DEBATE_OUT,
                0.8,
                1.2,
            )
        )
    return lines


def plan_optimize(auto: str | None = None) -> list[dict]:
    """`optimize` — GEPA metric evals (scorer + judge feedback on imperfect)."""
    auto = auto or cfg.GEPA_AUTO or "light"
    metric_calls = {
        "light": 100,
        "medium": 300,
        "heavy": 1000,
    }.get(auto, 100)
    # Each metric call: 1 scorer eval; imperfect -> 1 judge LLM feedback.
    scorer_evals = metric_calls
    judge_calls = int(metric_calls * 0.7)  # ~70% imperfect -> feedback
    lines = [
        _line(cfg.PRIMARY_LM, scorer_evals, TOKENS_SCORE_IN, TOKENS_SCORE_OUT, 0.8, 1.2),
        _line(cfg.REFLECTION_LM, judge_calls, TOKENS_JUDGE_IN, TOKENS_JUDGE_OUT, 0.8, 1.2),
    ]
    return lines


def plan_track2(n_iter: int | None = None) -> list[dict]:
    """`track2` — ReActV2 agent loop (1-2 LLM calls per iteration)."""
    iters = n_iter or cfg.TRACK2_MAX_ITERS
    calls = int(iters * 1.5)  # ~1.5 calls/iteration avg (reason + tool use)
    return [
        _line(cfg.PRIMARY_LM, calls, TOKENS_AGENT_IN, TOKENS_AGENT_OUT, 0.8, 1.2),
    ]


def plan_evaluate(n_examples: int = 12) -> list[dict]:
    """`evaluate` — judge score per devset example."""
    return [
        _line(cfg.PRIMARY_LM, n_examples, TOKENS_SCORE_IN, TOKENS_SCORE_OUT, 0.8, 1.2),
    ]


def plan_full_runs(n_score_runs: int = 3, n_panel_runs: int = 2) -> list[dict]:
    """Full-run spend plan: repeated score/panel + 1 optimize + track2."""
    lines = []
    for _ in range(n_score_runs):
        lines.extend(plan_score())
    for _ in range(n_panel_runs):
        lines.extend(plan_panel())
    lines.extend(plan_optimize())
    lines.extend(plan_track2())
    return lines


# ── CLI-facing summary ──────────────────────────────────────────────────────


def print_plan(plan_name: str, lines: list[dict], total_calls: int | None = None) -> None:
    """Print a plan with per-line brackets and a total low/high."""
    low = sum(line["low"] for line in lines)
    high = sum(line["high"] for line in lines)
    total_calls = total_calls or sum(line["calls"] for line in lines)
    print(f"\n[{plan_name}]")
    for line in lines:
        print(_fmt_line(line))
    print(
        f"  {'TOTAL':<42s} {total_calls:>5d} calls "
        f"{sum(line['tokens_in'] for line in lines):>9,} in / "
        f"{sum(line['tokens_out'] for line in lines):>6,} out  "
        f"{_fmt_bracket(low, high)}"
    )
    print(f"  → estimated spend: {_fmt_bracket(low, high)}")


def cmd_cost_estimate(args) -> None:
    """Dry-run budget plan for every pipeline command (no API calls made)."""
    print("MVA pipeline — cost dry-run (no API calls made; prices 2026-08-30)")
    print(f"Primary: {cfg.PRIMARY_LM}  ·  Panel: {len(cfg.PANEL)} members")
    print(f"GEPA auto={cfg.GEPA_AUTO}  ·  Track2 max_iters={cfg.TRACK2_MAX_ITERS}")

    print_plan(
        f"score   (max-candidates={args.max_candidates}, max-pairs={args.max_pairs}"
        f"{', +RLM' if args.rlm else ''})",
        plan_score(args.max_candidates, args.max_pairs, rlm=args.rlm),
    )
    print_plan(
        f"panel   (max-candidates={args.max_candidates}, max-pairs={args.max_pairs})",
        plan_panel(args.max_candidates, args.max_pairs),
    )
    print_plan(
        f"optimize (GEPA {cfg.GEPA_AUTO or 'light'})",
        plan_optimize(),
    )
    print_plan(
        f"evaluate (devset={args.evaluate_examples})",
        plan_evaluate(args.evaluate_examples),
    )
    print_plan(
        f"track2  (max_iters={cfg.TRACK2_MAX_ITERS})",
        plan_track2(),
    )

    full = plan_full_runs()
    print_plan("FULL HACKATHON (3× score + 2× panel + 1× optimize + 1× track2)", full)


def add_cost_subcommand(sub, parser_root) -> None:
    """Register the `cost` subcommand."""
    sp = sub.add_parser("cost", help="dry-run budget estimate (no API calls)")
    sp.add_argument("--max-candidates", type=int, default=500)
    sp.add_argument("--max-pairs", type=int, default=10)
    sp.add_argument("--rlm", action="store_true", help="include the RLM scorer in the plan")
    sp.add_argument("--evaluate-examples", type=int, default=12)
    sp.set_defaults(func=cmd_cost_estimate)
