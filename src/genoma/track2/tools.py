"""Track 2 — tools for the ReActV2 drug-repurposing agent.

Tools are plain callables; ReActV2 wraps them into dspy.Tool. They hit
free/public APIs (NCBI E-utilities, OpenFDA). Fail-fast: any network/parse
error propagates — no silent error-note substitution, so a failed tool call
is surfaced loudly instead of quietly degrading the agent's evidence.

We also export explicitly-wrapped ``dspy.Tool`` objects (with arg
descriptions) because ReActV2's native-function-calling schema benefits
from precise parameter docs: ReActV2 accepts ``list[Callable | Tool]``, and
explicit Tool objects keep the schema richer than auto-inferred
names/descriptions.
"""

from __future__ import annotations

import json
import urllib.parse

import dspy

from genoma.net import urlopen_with_retry


def pubmed_search(query: str, max_results: int = 5) -> str:
    """Search PubMed via NCBI E-utilities (no API key required).

    Returns a compact JSON string of {title, pubdate, source, pmid}.
    """
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    term = urllib.parse.quote(query)
    with urlopen_with_retry(
        f"{base}esearch.fcgi?db=pubmed&term={term}&retmax={max_results}&retmode=json",
        timeout=20,
    ) as r:
        ids = json.load(r)["esearchresult"].get("idlist", [])
    if not ids:
        return json.dumps({"query": query, "results": []})
    with urlopen_with_retry(
        f"{base}esummary.fcgi?db=pubmed&id={','.join(ids)}&retmode=json",
        timeout=20,
    ) as r:
        docs = json.load(r)["result"]
    out = []
    for pid in ids:
        d = docs.get(pid, {})
        out.append(
            {
                "pmid": pid,
                "title": d.get("title", ""),
                "source": d.get("source", ""),
                "pubdate": d.get("pubdate", ""),
            }
        )
    return json.dumps({"query": query, "results": out})


def drugbank_lookup(drug: str) -> str:
    """Look up a drug's mechanism of action / targets via OpenFDA.

    Uses the public drug/label endpoint (key-free, rate-limited):
    https://api.fda.gov/drug/label.json?search=openfda.brand_name:"X"
    Returns structured JSON: generic/brand name, mechanism of action,
    pharmacologic class, indications.
    """
    term = urllib.parse.quote(f'openfda.brand_name:"{drug}"')
    url = f"https://api.fda.gov/drug/label.json?search={term}&limit=1"
    with urlopen_with_retry(url, timeout=20) as r:
        data = json.load(r)
    results = data.get("results")
    if not results:
        return json.dumps({"drug": drug, "results": [], "note": "no label found"})
    label = results[0]
    openfda = label.get("openfda", {})
    return json.dumps(
        {
            "drug": drug,
            "generic_name": (openfda.get("generic_name") or [])[:3],
            "brand_name": (openfda.get("brand_name") or [])[:3],
            "mechanism_of_action": (label.get("mechanism_of_action") or [])[:3],
            "pharm_class": (label.get("pharm_class_epc") or [])[:3],
            "indications": (label.get("indications_and_usage") or [""])[0][:600],
        }
    )


def clinvar_lookup(term: str) -> str:
    """Query ClinVar via NCBI E-utilities for a gene/variant.

    Uses the free clinvar ESearch/ESummary endpoints.
    """
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    term = urllib.parse.quote(term)
    with urlopen_with_retry(
        f"{base}esearch.fcgi?db=clinvar&term={term}&retmax=10&retmode=json",
        timeout=20,
    ) as r:
        ids = json.load(r)["esearchresult"].get("idlist", [])
    if not ids:
        return json.dumps({"term": term, "results": []})
    with urlopen_with_retry(
        f"{base}esummary.fcgi?db=clinvar&id={','.join(ids)}&retmode=json",
        timeout=20,
    ) as r:
        docs = json.load(r)["result"]
    out = []
    for cid in ids:
        d = docs.get(cid, {})
        out.append(
            {
                "clinvar_id": cid,
                "title": d.get("title", ""),
                "clinical_significance": d.get("clinical_significance", {}).get("description", ""),
            }
        )
    return json.dumps({"term": term, "results": out})


# Explicit dspy.Tool wrapping: richer arg descriptions for native function
# calling (ReActV2 builds its tool schema from name/desc/args/arg_desc).
TRACK2_TOOLS_WRAPPED = [
    dspy.Tool(
        pubmed_search,
        name="pubmed_search",
        desc="Search PubMed (NCBI E-utilities) for publications matching a query.",
        arg_desc={
            "query": "search terms (e.g. 'BUB1B mitotic checkpoint')",
            "max_results": "max PMIDs to return (default 5)",
        },
    ),
    dspy.Tool(
        drugbank_lookup,
        name="drugbank_lookup",
        desc="Look up an approved drug's mechanism of action / targets via OpenFDA.",
        arg_desc={"drug": "brand or generic drug name (e.g. 'dasatinib')"},
    ),
    dspy.Tool(
        clinvar_lookup,
        name="clinvar_lookup",
        desc="Query ClinVar (NCBI E-utilities) for a gene or variant's clinical significance.",
        arg_desc={"term": "gene symbol or HGVS-style variant term"},
    ),
]
