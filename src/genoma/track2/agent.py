"""Track 2 — drug repurposing via a ReActV2 tool-using agent.

ReActV2 (dspy 3.3.1 experimental) is the current tool-agent primitive:
structured History, native function calling, parallel tool calls, and a
`submit` tool that carries typed outputs. We enable native function
calling via ChatAdapter(use_native_function_calling=True).
"""

from __future__ import annotations

import dspy
from pydantic import BaseModel, Field

from genoma import config as cfg
from genoma.lm import native_tool_adapter, primary_lm
from genoma.track2.tools import TRACK2_TOOLS_WRAPPED


class DrugCandidate(BaseModel):
    name: str
    mechanism: str = Field(description="proposed mechanism of action")
    rationale: str = Field(description="how it targets the variant-driven pathway")
    evidence: str = Field(description="supporting evidence / citations")
    risk: str = Field(description="key caveats / unknowns")


class DrugRepurposingSignature(dspy.Signature):
    """You are a clinical-pharmacology researcher. Given the variant-driven
    molecular mechanism in this MVA proband and the phenotype, propose
    existing, market-approved medications that plausibly act on the disrupted
    pathway. Use the tools to gather evidence. These are hypotheses for
    follow-up investigation — not claims that a medicine works.

    Example mechanisms in MVA: spindle-assembly-checkpoint failure,
    chromosome missegregation, centrosome defects. Consider whether any
    approved drug modulates mitotic checkpoint kinases (e.g. MPS1/TTK),
    PLK1, Aurora kinases, or downstream consequences.

    Therapeutic window (critical for expert review): the patient is a child
    with growth delay and systemic chromosomal instability. AVOID proposing
    broadly cytotoxic spindle poisons (taxanes, vinca alkaloids) or other
    non-selective mitotic inhibitors — they would worsen aneuploidy and
    toxicity in non-cancerous tissues. PRIORITIZE agents exploiting
    aneuploid-cell-selective vulnerabilities (proteotoxic-stress response,
    HSP90 inhibition, autophagy modulation) or compounds that stabilize
    centriole/centrosome function with tolerable pediatric safety profiles.
    State explicitly, per candidate, why the benefit-risk balance favors it
    in THIS proband."""

    mechanism: str = dspy.InputField(desc="characterized variant-driven mechanism")
    gene: str = dspy.InputField(desc="gene affected")
    hpo_terms: list[str] = dspy.InputField(desc="proband phenotype terms")

    drug_candidates: list[DrugCandidate] = dspy.OutputField(
        desc="ranked repurposing candidates with mechanism, rationale, evidence, risk"
    )


class Track2Agent(dspy.Module):
    def __init__(self, tools: list | None = None, max_iters: int | None = None) -> None:
        super().__init__()
        self.agent = dspy.ReActV2(
            DrugRepurposingSignature,
            tools=tools or TRACK2_TOOLS_WRAPPED,
            max_iters=max_iters or cfg.TRACK2_MAX_ITERS,
        )

    def forward(self, mechanism: str, gene: str, hpo_terms: list[str]) -> dspy.Prediction:
        # ReActV2 drives native function calling; ChatAdapter flag on.
        with dspy.context(lm=primary_lm(), adapter=native_tool_adapter()):
            out = self.agent(mechanism=mechanism, gene=gene, hpo_terms=hpo_terms)
        history = getattr(out, "history", None)
        return dspy.Prediction(
            drug_candidates=out.drug_candidates,
            history=history,
            termination_reason=getattr(out, "termination_reason", None),
            stall_diagnosis=diagnose_tool_loop(history),
        )


def diagnose_tool_loop(history) -> str:
    """Post-hoc stall diagnosis over the recorded agent history.

    ReActV2's own loop isn't safely instrumentable from outside, so this is
    a lightweight progress gate *over the recorded history*: if the agent
    repeated the same tool call verbatim or never varied its tools, it likely
    burned its step budget on a dead end. The diagnosis is surfaced in the
    Track 2 report so a human reviewer sees the limitation instead of
    trusting a low-evidence answer. Returns "" when the history shows no
    stall. Defensive over the opaque history shape (dicts or strings).
    """
    if not history:
        return ""
    calls: list[str] = []
    for step in history:
        if isinstance(step, dict):
            for key in ("tool_call", "tool", "action"):
                val = step.get(key)
                if isinstance(val, str) and val.strip():
                    calls.append(val.strip())
                    break
        elif isinstance(step, str) and ("(" in step or ":" in step):
            calls.append(step)
    if not calls:
        return ""
    repeats = sum(1 for a, b in zip(calls, calls[1:], strict=False) if a == b)
    distinct = len(set(calls))
    reasons: list[str] = []
    if repeats >= 2:
        reasons.append(f"repeated the same tool call {repeats} time(s)")
    if distinct <= 1 and len(calls) >= 3:
        reasons.append("never varied tools across steps")
    if len(calls) >= 6 and distinct <= 2:
        reasons.append("narrow tool usage for the step budget")
    if not reasons:
        return ""
    return "Possible stall: " + "; ".join(reasons) + "."
