"""Phen2Gene: HPO-term -> gene ranking via the REST API (fail-fast).

Primary path is the Phen2Gene REST API (https://phen2gene.wglab.org/api —
no auth, no install).

Design: fail fast. Any network/API/parse error propagates — there is no
silent fallback. Use ``_fallback_ranking()`` explicitly if an offline prior
is wanted instead.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

from genoma.data.annotate import KNOWN_MVA_GENES
from genoma.net import urlopen_with_retry

_API_URL = "https://phen2gene.wglab.org/api"


def run_phen2gene(
    hpo_terms: list[str],
    *,
    timeout: int = 60,
) -> dict[str, float]:
    """Rank genes against the HPO terms via the Phen2Gene REST API.

    Fails fast: any network/API/parse error propagates — there is no silent
    fallback. Use ``_fallback_ranking()`` explicitly if an offline prior is
    wanted instead.
    """
    hpo_ids = [t.split()[0].strip().rstrip(",") for t in hpo_terms if t.strip()]
    hpo_ids = [h for h in hpo_ids if h.upper().startswith("HP:")]
    if not hpo_ids:
        raise ValueError("No valid HPO terms to rank against. Phen2Gene requires HP: identifiers.")
    hpo_ids = sorted(set(hpo_ids))  # canonical order = stable cache key

    # Local cache: same HPO set -> same ranking; repeated runs are free.
    from genoma.data.annot_cache import cache_get, cache_set

    key = "phen2gene:" + ";".join(hpo_ids)
    cached = cache_get(key)
    if cached is not None:
        return cached
    scores = _run_api(hpo_ids, timeout)
    cache_set(key, scores)
    return scores


def _run_api(hpo_ids: list[str], timeout: int) -> dict[str, float]:
    query = urllib.parse.urlencode({"HPO_list": ";".join(hpo_ids), "weight_model": "sk"})
    with urlopen_with_retry(f"{_API_URL}?{query}", timeout=timeout) as r:
        data = json.load(r)
    results = data.get("results", [])
    if not results:
        raise RuntimeError("Phen2Gene API returned no results")
    scores: dict[str, float] = {}
    for item in results:
        gene = str(item.get("Gene", "")).strip().upper()
        score = float(item.get("Score", 0.0))  # fail fast on malformed score
        if gene:
            scores[gene] = max(scores.get(gene, 0.0), score)
    if not scores:
        raise RuntimeError("Phen2Gene API results parsed empty")
    return scores  # already ~normalized (top gene = 1.0)


def _fallback_ranking() -> dict[str, float]:
    """Curated MVA prior: phenotype-literature-weighted gene order.

    BUB1B is the archetypal MVA gene (MVA1) with the deepest phenotype
    literature; the rest follow established case counts. The gene set is the
    SSOT ``KNOWN_MVA_GENES`` (annotate.py); the weighted ORDER lives here.
    """
    order = [
        "BUB1B",
        "CEP57",
        "TRIP13",
        "MAD1L1",
        "MAD2L1BP",
        "CEP192",
    ]
    assert set(order) == KNOWN_MVA_GENES, "curated order must match KNOWN_MVA_GENES"
    n = len(order)
    return {g: round((n - i) / n, 4) for i, g in enumerate(order)}


def rank_mva_genes(hpo_terms: list[str]) -> dict[str, float]:
    """Phenotype ranking restricted to the six MVA genes (1.0 = best fit).

    Fail-fast: if the API returns no hits for the six MVA genes the caller
    should decide how to proceed — no silent substitution.
    """
    scores = run_phen2gene(hpo_terms)
    mva = {g: scores.get(g, 0.0) for g in KNOWN_MVA_GENES}
    mx = max(mva.values())
    if mx <= 0:
        raise RuntimeError("Phen2Gene returned no scores for any of the six MVA genes.")
    return {g: s / mx for g, s in mva.items()}
