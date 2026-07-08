# Research Problem

## Motivation

Scientific literature screening is difficult when the user is still learning the
field. A novice often knows the phenomenon they care about, but not the expert
vocabulary, canonical mechanisms, benchmark methods, exclusion boundaries, or
provider-specific search syntax needed to retrieve the right papers.

`search_paper` studies the following problem:

```text
How can a human-centric multi-agent LLM system transform noisy novice research
questions into evidence-grounded, domain-aware, feedback-adaptive scientific
literature screening artifacts?
```

The current v9 baseline is deterministic and rule-controlled. It uses bounded
agents, explicit artifacts, span validation, domain guardrails, and transparent
ranking. LLMs are treated as a controlled enhancement layer for understanding or
criticizing user intent, not as the authority that decides evidence validity,
include/exclude labels, reading priority, final scores, or domain decisions.

## Why Novice Literature Screening Is Hard

Novice literature screening is not just keyword search. It has several recurring
failure modes:

- The user question is a noisy expression of research intent, not a provider-ready query.
- The user may write in Chinese, English, or mixed language while providers need English scholarly terms.
- Important concepts may be implied rather than named, such as lithium context in a Chinese SEI battery question.
- Acronyms are ambiguous across fields, for example SEI, OER, MOF, AI, and PFM.
- A query can retrieve papers that match surface terms but belong to the wrong domain.
- A paper can be relevant enough to keep, but not important enough to be a first-read `must_read` paper.
- Abstract-only evidence is limited and must not be inflated into unsupported claims.
- A novice user needs a reading path and explanation, not only a long ranked CSV.

The system therefore treats the user question as evidence about intent. It must
repair and operationalize that intent while keeping every decision inspectable.

## Formal Problem Formulation

Let:

- `q` be a noisy novice research question.
- `S` be optional seed papers supplied as DOI, title, Semantic Scholar ID, OpenAlex ID, or CSV rows.
- `H` be optional user feedback history over previous include, exclude, or uncertain decisions.
- `B` be the provider/query budget, including maximum papers per query and provider limits.
- `P` be the set of scholarly providers or import sources, such as OpenAlex, Semantic Scholar, BibTeX, RIS, or CSV.
- `U` be the user profile or expertise level, for example novice, domain learner, or expert reviewer.

The system estimates a structured intent representation:

```text
I_hat = IntentRepair(q, S, H, U)
```

It then builds a search contract and query-family plan:

```text
K = SearchContract(I_hat)
Q = QueryFamilyPlanner(I_hat, K, B, P)
```

For each provider or import source, it retrieves or imports candidates:

```text
C_raw = Retrieve(Q, P, B) union Import(S)
C = Deduplicate(C_raw)
```

For each candidate paper `p` in `C`, the system computes:

```text
E_p = EvidenceExtraction(p, I_hat)
V_p = SpanValidation(E_p, abstract_p)
D_p = DomainGuardrail(p, K)
A_p = RequiredGroupCoverage(p, K)
R_p = Rank(p, I_hat, V_p, D_p, A_p, H)
```

The final system output is:

```text
Y = (Q, C, E, D, R, G, pi)
```

where `pi` is the next-step search or feedback policy exposed to the user.

## Inputs

The baseline input variables are:

- `q`: noisy novice research question.
- `S`: optional seed papers.
- `H`: optional user feedback history.
- `B`: provider/query budget.
- `P`: scholarly providers and local import sources.
- `U`: user profile or expertise level.

Concrete repository inputs include:

- CLI or Streamlit question text.
- Provider choices and `max_per_query` budget.
- Optional BibTeX, RIS, or CSV library exports.
- Optional seed-paper entries.
- Optional feedback CSV.
- Optional LLM configuration, which must remain advisory and non-authoritative.

## Outputs

The target outputs are:

- `Q`: query family plan and provider-specific query plan.
- `C`: normalized and deduplicated candidate papers.
- `E`: evidence spans and span-validation diagnostics.
- `D`: domain guardrail decisions and context warnings.
- `R`: ranked reading list with scores, decisions, and reading priorities.
- `G`: user-facing report and reading path.
- `π`: next-step search/feedback policy.

Concrete repository artifacts include:

- `search_brief.json`, `search_contract.json`, `ambiguity_analysis.json`, and `question_refinement.json`.
- `planned_queries.json`, `query_families.json`, `query_provenance.json`, and query-repair diagnostics.
- Raw provider JSON, `merged_papers.csv`, and retrieval diagnostics.
- `evidence_table.csv`, `evidence_functions.json`, and span-validation fields.
- `domain_assessments.json`, `aspect_coverage.csv`, `screening_decisions.*`, and `ranked_papers.csv`.
- `paper_roles.json`, `research_tensions.json`, `paper_cards.md`, `reading_path.md`, `report.md`, and `user_report.md` when generated.
- `evaluation.json`, `exploration_quality.json`, `agent_trace.json`, and `run_events.jsonl`.

## Objective

The system objective is to maximize:

- Intent relevance: results match the repaired research question, not just raw keywords.
- Aspect coverage: required concept groups appear in the candidate set and top results.
- Evidence grounding: claims trace to title/abstract spans with explicit support level.
- Diversity: the reading list covers mechanisms, methods, reviews, applications, and frontier papers when appropriate.
- User preference fit: feedback can adjust future ranking and search directions.

The system objective is also to minimize:

- False positives and wrong-domain drift.
- Query cost and provider waste.
- Clarification burden on novice users.
- Ungrounded claims or evidence hallucination.
- Overloaded reports that are complete but not actionable.

The design principle is:

```text
LLM understands the user.
Rules protect the evidence.
Ranking organizes the reading path.
Report explains the reasoning.
```

## Agent Roles

The repository uses bounded agents and artifact writers. In the v9 deterministic
baseline, these are mostly rule-controlled modules rather than independent LLM
agents.

- NoviceIntentInterpreter / Intent Repair: interprets the raw user question, repairs novice phrasing, and normalizes multilingual concepts.
- ExpertResearchIntent / Structured Concepts: turns repaired intent into required concepts, exclusions, target context, and success criteria.
- DomainRouter: selects lightweight domain knowledge and domain-boundary hints.
- SearchContract: records required groups, target context groups, negative context groups, and ambiguity decisions.
- QueryFamilyPlanner: creates multiple search routes for different research lenses.
- QueryRepair and QueryCritic diagnostics: audit candidate queries, repair overbroad or malformed queries when enabled, and record whether repair changed the final plan.
- Provider retrieval: queries OpenAlex and Semantic Scholar, or imports local library records.
- Evidence extraction and span validation: extracts candidate evidence and verifies whether it is grounded in the abstract.
- DomainGuardrail: labels papers as in-scope, borderline/peripheral, or out-of-scope using the search contract.
- PaperRoleClassifier: assigns reading roles such as review, method, mechanism, evaluation, background, or frontier when supported.
- IntentCentrality and group-coverage ranking: organize papers by task usefulness, not only provider score or citation count.
- Context-aware reading priority: separates `include` from first-read `must_read` papers.
- Reading path and user report: turn ranked artifacts into a novice-friendly reading plan and caveated explanation.
- Human feedback: records include/exclude/uncertain signals and supports future preference adaptation.

## Failure Modes

The system should expose failures instead of hiding them behind a polished ranked
list.

- Intent over-repair: the system imposes a narrower expert intent than the user asked for.
- Intent under-repair: novice wording is kept too literally and expert anchors are missed.
- Query-family degradation: distinct research routes collapse into generic keyword queries.
- Query repair ambiguity: final queries may look unchanged if upstream sanitizers already cleaned them.
- Provider instability: API errors, rate limits, missing keys, or metadata gaps distort retrieval.
- Domain drift: papers match surface terms but belong to a forbidden or peripheral domain.
- Target-context failure: lithium-specific battery tasks retrieve sodium, potassium, zinc, or other non-target systems as core papers.
- Evidence inflation: weak keyword overlap is mistaken for strict support.
- Ranking miscalibration: `include` becomes equivalent to `must_read` or no papers are promoted to `must_read` when target context is not required.
- Reading-path leakage: excluded or out-of-scope papers appear in recommended reading sections.
- Role misclassification: useful top papers are left unclassified or assigned the wrong reading role.
- Novice overload: reports list artifacts without a clear first reading path.

## Evaluation Overview

Evaluation should compare the deterministic full system against pilot ablations
and future LLM-enhanced variants. Current evaluation is diagnostic, not a final
formal ablation study.

Plan-only evaluation checks:

- QueryFamily application and coverage.
- Query quality score.
- Anchor coverage.
- Single-acronym, overbroad, repeated-phrase, single-axis, and weak-query counts.
- Plan-only retrieval status and skipped research-gap generation.

Full-run evaluation checks:

- Provider success rate, raw retrieved count, merged count, and duplicate ratio.
- Domain false positives in top-10, top-20, include, and must-read sets.
- Required group coverage and intent centrality in top results.
- Evidence support and span validation.
- Must-read count and include count calibration.
- Reading-path diagnostics for exclude, out-of-scope, duplicate, and negative-context leakage.
- Whether target context is required for reading priority.

Future LLM-enhanced evaluation should test whether LLMIntentFrameEnhancer or
LLMQueryPlanCritic improves intent understanding or query quality without taking
over rule-owned decisions.
