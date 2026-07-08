import json
import re
import sys
from pathlib import Path

BAD_STANDALONE = {
    "ex situ",
    '"ex situ"',
    "in situ",
    '"in situ"',
    "failure mechanism",
    '"failure mechanism"',
    "theoretical mechanism",
    '"theoretical mechanism"',
    "experimental characterization",
    '"experimental characterization"',
    "characterization method",
    '"characterization method"',
    "lithium-ion battery",
    '"lithium-ion battery"',
    "thin film",
    '"thin film"',
    "spin state",
    '"spin state"',
    "surface spin state",
    '"surface spin state"',
    "carbon dioxide",
    '"carbon dioxide"',
    "carrier recombination",
    '"carrier recombination"',
    "comparison",
    "human-in-the-loop",
}

SINGLE_ACRONYM = re.compile(r'^\+?"?(SEI|OER|PFM|SHG|LLM|MOF)"?$', re.I)

def load(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def flatten_strings(x):
    out = []
    if isinstance(x, str):
        out.append(x)
    elif isinstance(x, list):
        for i in x:
            out += flatten_strings(i)
    elif isinstance(x, dict):
        for v in x.values():
            out += flatten_strings(v)
    return out

def get_final_queries(run):
    qp = load(run / "query_provenance.json")
    planned = load(run / "planned_queries.json")

    candidates = []

    for key in [
        "final_openalex_queries",
        "final_semantic_scholar_queries",
        "final_provider_queries",
    ]:
        if key in qp:
            candidates += flatten_strings(qp[key])
        if key in planned:
            candidates += flatten_strings(planned[key])

    # Fallback: read query_plan only if explicit final fields are absent.
    if not candidates:
        qplan = planned.get("query_plan", {})
        for key in ["openalex_queries", "semantic_scholar_queries"]:
            candidates += flatten_strings(qplan.get(key, []))

    clean = []
    for q in candidates:
        q = str(q).strip()
        if q and q not in clean:
            clean.append(q)
    return clean

def text_blob(run):
    objs = [
        load(run / "planned_queries.json"),
        load(run / "query_provenance.json"),
        load(run / "agent_trace.json"),
        load(run / "search_contract.json"),
        load(run / "query_repair_suggestions.json"),
        load(run / "exploration_quality.json"),
    ]
    return json.dumps(objs, ensure_ascii=False).lower()

def has_any(q, terms):
    ql = q.lower()
    return any(t.lower() in ql for t in terms)

def check_sei(queries):
    failures = []
    for q in queries:
        if q.lower() in BAD_STANDALONE or SINGLE_ACRONYM.match(q):
            failures.append(f"bad standalone SEI query: {q}")

    anchored = [
        q for q in queries
        if has_any(q, ["SEI", "solid electrolyte interphase"])
        and has_any(q, ["lithium-ion", "battery", "anode"])
    ]
    if len(anchored) < max(5, len(queries) // 2):
        failures.append("too few SEI queries anchored by SEI/solid electrolyte interphase + battery/anode context")
    return failures

def check_oer(queries):
    failures = []
    for q in queries:
        if q.lower() in BAD_STANDALONE or SINGLE_ACRONYM.match(q):
            failures.append(f"bad standalone OER query: {q}")

    anchored = [
        q for q in queries
        if has_any(q, ["OER", "oxygen evolution", "water oxidation"])
        and has_any(q, ["spin", "electronic", "orbital"])
        and has_any(q, ["catalyst", "oxide", "electrocatalyst", "transition metal"])
    ]
    if len(anchored) < 4:
        failures.append("too few OER queries anchored by OER + spin/electronic + catalyst/oxide context")
    return failures

def check_mof(queries):
    failures = []
    joined = " | ".join(queries).lower()
    if not any("mof" in q.lower() and any(x in q.lower() for x in ["co2", "carbon dioxide", "capture", "adsorption"]) for q in queries):
        failures.append("MOF queries not anchored by MOF + CO2/capture/adsorption")
    for term in ["pore", "functional", "water stability", "adsorption performance"]:
        if term not in joined:
            failures.append(f"MOF aspect missing: {term}")
    return failures

def check_perovskite(queries, blob):
    failures = []
    if "battery" in blob:
        failures.append("battery appears in perovskite solar run; check cross-domain injection")
    if not any("perovskite solar" in q.lower() and any(x in q.lower() for x in ["defect", "passivation", "recombination"]) for q in queries):
        failures.append("perovskite queries not anchored by perovskite solar cell + defect/passivation/recombination")
    return failures

def check_thin_film(queries, blob):
    failures = []
    for q in queries:
        if q.lower() in {"thin film", '"thin film"', "comparison ald", "ald comparison"}:
            failures.append(f"bad standalone thin-film query: {q}")
    if not any("thin film deposition" in q.lower() and any(x in q for x in ["ALD", "PLD", "sputtering", "CVD"]) for q in queries):
        failures.append("thin-film queries lack thin film deposition + methods combination")
    for bad in ["PFM", "SHG", "BaTiO3", "depolarization"]:
        if bad.lower() in blob:
            failures.append(f"ferroelectric term injected into thin-film run: {bad}")
    return failures

def check_ai(queries, blob):
    failures = []
    joined = " | ".join(queries).lower()
    for needed in ["llm", "literature screening"]:
        if needed not in joined:
            failures.append(f"AI screening missing core term: {needed}")
    for bad in ["sei", "oer", "pfm", "shg", "cr2o3", "spleem"]:
        if bad in blob:
            failures.append(f"unrelated domain term injected into AI screening run: {bad}")
    return failures

def check_run(run_dir):
    run = Path(run_dir)
    queries = get_final_queries(run)
    blob = text_blob(run)

    failures = []

    if "unsupported_domain" in blob and "skipped" in blob:
        failures.append("still has unsupported_domain + skipped")

    if '"query_family_applied": false' in blob:
        failures.append("query_family_applied=false")

    if len(queries) < 8:
        failures.append(f"final provider query count too low: {len(queries)}")

    for q in queries:
        if SINGLE_ACRONYM.match(q):
            failures.append(f"single acronym final query: {q}")

    name = run.name.lower()
    if "sei" in name:
        failures += check_sei(queries)
    elif "oer" in name:
        failures += check_oer(queries)
    elif "mof" in name:
        failures += check_mof(queries)
    elif "perovskite" in name:
        failures += check_perovskite(queries, blob)
    elif "thin_film" in name:
        failures += check_thin_film(queries, blob)
    elif "ai_screening" in name:
        failures += check_ai(queries, blob)

    print(f"\n=== {run} ===")
    print(f"final queries: {len(queries)}")
    for q in queries[:25]:
        print(f"  - {q}")

    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        return False

    print("PASS")
    return True

if __name__ == "__main__":
    ok = True
    for arg in sys.argv[1:]:
        ok = check_run(arg) and ok
    sys.exit(0 if ok else 1)

