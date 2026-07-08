# Research Problem

## Motivation

Scientific literature screening is difficult for novice scientific users because
they often do not yet know the expert vocabulary, canonical papers, methods,
materials, exclusion boundaries, or provider-specific search conventions of a
field. A normal keyword search assumes that the user can already express an
expert query. Early-stage research exploration often violates that assumption.

This project studies evidence-grounded literature screening under imperfect
novice intent. A user may submit a question that is vague, over-broad,
over-narrow, multilingual, partially wrong, missing key concepts, or phrased as
a desired research direction rather than a precise search query. The system
therefore should not treat the raw user question as the final search query. It
should infer and repair the research intent, expose the repair for inspection,
retrieve or import candidate papers, ground claims in abstracts, rank papers
transparently, accept human feedback, and write auditable artifacts.

The central research question is:

```text
How can a human-centric multi-agent AI system transform incomplete, ambiguous,
or partially incorrect novice research questions into verifiable, domain-aware,
feedback-adaptive literature screening decisions?
```

## Formal Problem Formulation

Let `x` be the raw user question. The question may be incomplete, ambiguous,
multilingual, or partially incorrect. The system assumes that there is a latent
expert-level research intent `z*`, but `z*` is not directly observable.

The system constructs an auditable approximation of that intent:

```text
z_hat = intent_repair(x, h, d)
```

where:

- `x` is the raw user question.
- `h` is optional human feedback history.
- `d` is optional domain knowledge, such as domain packs, synonym lists,
  false-positive terms, method terms, or seed-paper hints.
- `z_hat` is represented through structured artifacts such as `SearchBrief`,
  `search_contract.json`, ambiguity analysis, concept maps, query families, and
  required aspects.

From `z_hat`, the system creates provider-aware query plans:

```text
Q = query_planning(z_hat, d)
```

Each query is more than a string: it should have a source, provider, purpose,
and relation to the repaired research intent. The current implementation keeps
legacy `planned_queries.json` as the default retrieval plan and keeps
query-family retrieval optional.

For each enabled provider or import source `r`, the retriever returns candidate
papers:

```text
C_r = retrieve_or_import(Q_r, budget_r)
C = deduplicate(union(C_r))
```

For each paper `p` in the merged candidate set, the system computes:

```text
e_p = extract_evidence(p, z_hat)
v_p = verify_span(e_p, abstract_p)
g_p = assess_domain_fit(p, z_hat, d)
role_p = classify_research_role(p, Q, z_hat)
s_p = rank(p, z_hat, e_p, v_p, g_p, role_p, h)
```

The final output is a structured screening artifact:

```text
Y = (ranked_papers, evidence, domain_assessments, roles, diagnostics, trace, report)
```

The research problem is to design a bounded multi-agent policy that maximizes
intent alignment, evidence grounding, domain precision, aspect coverage, ranking
usefulness, feedback adaptation, and interpretability under uncertainty.

## Inputs And Outputs

Primary input:

- A natural-language research question from the user.
- The question may mix Chinese and English, contain vague concepts, include
  partial seed-paper hints, or express the user's goal rather than an expert
  query.

Optional inputs:

- Human include, exclude, or uncertain feedback.
- Existing literature-library exports in BibTeX, RIS, or CSV format.
- Seed papers supplied by DOI, title, Semantic Scholar ID, OpenAlex ID, or CSV.
- Domain packs with terms, synonyms, methods, materials, applications, and
  false-positive terms.
- Provider configuration, retrieval budgets, ranking weights, and optional LLM
  modes.

Core outputs:

- Repaired research intent: `search_brief.json`, `search_contract.json`,
  `ambiguity_analysis.json`, and `question_refinement.json`.
- Query planning artifacts: `planned_queries.json`, `concept_map.json`,
  `query_families.json`, `seed_hints.json`, and `query_provenance.json`.
- Candidate artifacts: raw provider result JSON files, imported-paper tables,
  citation-expansion tables, retrieval paths, and `merged_papers.csv`.
- Evidence artifacts: `evidence_table.csv`, `evidence_functions.json`, span
  validation fields, support levels, and paper evidence cards.
- Screening and ranking artifacts: `domain_assessments.json`,
  `aspect_coverage.csv`, `screening_decisions.*`, score breakdowns, and ranked
  paper CSV files.
- Sensemaking artifacts: `paper_roles.json`, `research_tensions.json`,
  `method_comparison_matrix.*`, `research_gap_matrix.*`,
  `suggested_next_searches.*`, `paper_cards.md`, and `reading_path.md`.
- Audit artifacts: `evaluation.json`, `retrieval_diagnostics.json`,
  `query_pilot_diagnostics.json`, `query_repair_suggestions.json`,
  `agent_trace.json`, `run_events.jsonl`, and `report.md`.

## Objective

The system-level objective is to make literature screening useful and
trustworthy for novice users. In practical terms, a good run should:

- Align the repaired intent and ranked results with the user's research need.
- Ground claims about papers in verifiable abstract spans.
- Reduce off-topic drift through explicit domain boundaries.
- Cover the required conceptual aspects of the repaired question.
- Rank papers by usefulness for this research task, not only by citation count,
  recency, provider score, or keyword overlap.
- Preserve uncertainty instead of hiding it.
- Adapt ranking and next-search suggestions to human feedback.
- Provide enough traceability for a user to inspect how the system reached its
  decisions.

The system should penalize unsupported evidence, domain drift, opaque decisions,
and cognitively overloaded reports that are technically complete but difficult
for novice users to act on.

## Agent Roles

The repository implements the screening workflow as a set of bounded agents and
artifact writers. The exact implementation is intentionally lightweight and
mostly rule-based by default.

- Novice intent repair: interprets the raw user question as a `SearchBrief` and
  optional refined subquestions.
- Search contract and domain boundary construction: records intended domain,
  required concepts, excluded concepts, and ambiguous terms.
- Concept mapping: decomposes the repaired intent into research lenses and
  concept groups.
- Query planning and query-family explanation: creates provider-aware query
  plans and optional query-family rationale.
- Seed extraction: identifies DOI, title, arXiv, author, Semantic Scholar, or
  OpenAlex hints without automatically enabling citation expansion.
- Provider retrieval and import: retrieves from OpenAlex and Semantic Scholar or
  imports local BibTeX, RIS, and CSV records.
- Optional seed-paper snowballing: expands through Semantic Scholar references,
  citations, and recommendations only when explicitly enabled.
- Normalization and deduplication: merges provider/import records into normal
  `Paper` objects while preserving provenance.
- Evidence extraction: extracts claim-level evidence from title and abstract.
- Evidence span verification: checks exact or high-confidence fuzzy matches
  against the abstract and marks weak or invalid support.
- Aspect and domain assessment: records required-aspect coverage and
  in-scope/borderline/out-of-scope domain fit.
- Paper role and evidence-function classification: explains whether a paper is
  background, method, proof, theory, application, limitation, or frontier work.
- Ranking and feedback adaptation: computes transparent score breakdowns and
  optional feedback-adjusted rankings.
- Reporting and trace generation: writes CSV, JSON, Markdown, diagnostics, and
  user-facing reports.

## Failure Modes

The project should make failure modes visible instead of hiding them behind a
single ranked list.

- Intent over-repair: the system infers a more specific or different expert
  intent than the user meant.
- Intent under-repair: the system keeps novice wording too literally and misses
  expert terminology.
- Query-family collapse: distinct research directions collapse into one generic
  search route.
- Query over-expansion: broad query expansion creates noisy, off-topic
  retrieval.
- Provider bias or failure: provider semantics, metadata quality, rate limits,
  or API errors distort the candidate pool.
- Domain drift: papers match surface terms but belong to the wrong scientific
  domain.
- Over-strict filtering: useful interdisciplinary papers are demoted too
  aggressively.
- Evidence hallucination: generated evidence cannot be matched back to the
  abstract.
- Abstract-only evidence limits: relevant papers may not contain enough detail
  in abstracts to validate a specific claim.
- Ranking bias: citation count, recency, provider score, or keyword overlap
  dominates intent centrality.
- Paper-role misclassification: the reading path mislabels a paper's function in
  the research landscape.
- Feedback overfitting: a small number of feedback actions causes excessive
  ranking or query adaptation.
- Novice cognitive overload: the output is complete but too flat, long, or
  unstructured to help a new user decide what to read.

## Evaluation Overview

Evaluation should compare the full system with simpler baselines and ablations:

- Raw keyword search.
- Single-provider retrieval.
- Retrieval without intent repair.
- Retrieval without query-family explanation.
- Ranking without evidence span validation.
- Ranking without domain guardrails.
- Ranking without human feedback.
- Single-prompt LLM paper recommendation.

Useful metrics include:

- Retrieval and ranking: precision@k, recall@k, nDCG@k, MAP, top-k in-scope
  rate, and top-k aspect coverage.
- Evidence grounding: strict support rate, weak support rate, invalid evidence
  rate, span match confidence, and missing-abstract rate.
- Domain control: false-positive rate, out-of-scope rate, borderline recovery,
  and domain assessment accuracy.
- Feedback adaptation: before/after ranking delta, feedback acceptance rate,
  learned-term usefulness, and reduction in repeated off-topic results.
- Usability: whether a novice user can identify must-read papers, explain why a
  paper is relevant, trust the evidence trail, and form a concrete reading plan.

The working hypothesis is that novice intent repair, query-family planning,
evidence span validation, domain guardrails, role-aware ranking, and human
feedback together produce more useful and trustworthy literature screening
outputs than keyword search or single-prompt paper recommendation.
