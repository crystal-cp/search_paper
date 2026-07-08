import json
import re
import sys
from pathlib import Path

BAD_SINGLE = re.compile(r'^\+?"?(SEI|OER|PFM|SHG|LLM|sputtering)"?$', re.I)

def load_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def collect_queries(obj):
    queries = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if "quer" in k.lower() and isinstance(v, list):
                    for item in v:
                        if isinstance(item, str):
                            queries.append(item)
                        elif isinstance(item, dict):
                            for vv in item.values():
                                if isinstance(vv, str):
                                    queries.append(vv)
                walk(v)
        elif isinstance(x, list):
            for item in x:
                walk(item)

    walk(obj)
    return sorted(set(q.strip() for q in queries if q and q.strip()))

def check_run(run_dir):
    run = Path(run_dir)
    planned = load_json(run / "planned_queries.json")
    trace = load_json(run / "agent_trace.json")
    quality = load_json(run / "exploration_quality.json")
    repair = load_json(run / "query_repair_suggestions.json")

    queries = collect_queries(planned)
    text_blob = json.dumps([planned, trace, quality, repair], ensure_ascii=False).lower()

    failures = []

    if "unsupported_domain" in text_blob and "skipped" in text_blob:
        failures.append("ConceptMapper/QueryFamilyPlanner may still be skipped by unsupported_domain")

    if "query_family_applied" in text_blob and '"query_family_applied": false' in text_blob:
        failures.append("query_family_applied=false")

    if len(queries) < 8:
        failures.append(f"provider query count too low: {len(queries)}")

    single_bad = [q for q in queries if BAD_SINGLE.match(q)]
    if single_bad:
        failures.append(f"single acronym/short query found: {single_bad}")

    if '"concept_coverage": 0.0' in text_blob:
        failures.append("concept_coverage=0.0")

    if '"query_family_coverage": 0.0' in text_blob:
        failures.append("query_family_coverage=0.0")

    print(f"\n=== {run} ===")
    print(f"queries: {len(queries)}")
    for q in queries[:20]:
        print(f"  - {q}")

    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return False

    print("PASS basic checks")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/check_validation_outputs.py outputs/validation/<run_dir> ...")
        sys.exit(2)

    ok = True
    for d in sys.argv[1:]:
        ok = check_run(d) and ok

    sys.exit(0 if ok else 1)

