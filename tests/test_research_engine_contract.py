import json
from pathlib import Path

from lit_screening.agents.search_contract import SearchContractAgent
from lit_screening.agents.seed_extraction import SeedExtractionAgent
from lit_screening.agents.intent_repair import NoviceIntentInterpreter
from lit_screening.models import ExpertResearchIntent, IntentConcept, SearchBrief
from lit_screening.pipeline import plan_screening_queries


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_search_contract_uses_structured_concepts_not_natural_language_criteria():
    brief = SearchBrief(
        original_question="How do probes measure an effect?",
        refined_question="How do probes measure an effect?",
        search_intent="overview",
        user_goal="Find evidence papers.",
        inclusion_criteria=[
            "Studies discussing why the effect is important",
            "surface-sensitive evidence",
        ],
    )
    intent = ExpertResearchIntent(
        original_question=brief.original_question,
        user_is_novice=True,
        inferred_goal="Find evidence papers.",
        expert_rewritten_question="Find evidence papers about a target effect.",
        structured_concepts=[
            IntentConcept(
                term="target effect",
                category="property",
                source="user_text",
                confidence=0.9,
                activation_reason="Explicit user target.",
                query_role="must",
                should_use_in_provider_query=True,
            ),
            IntentConcept(
                term="candidate probe",
                category="method",
                source="domain_pack",
                confidence=0.82,
                activation_reason="Probe wording activates methods.",
                query_role="optional",
                should_use_in_provider_query=True,
            ),
            IntentConcept(
                term="broad importance",
                category="motivation",
                source="user_text",
                confidence=0.5,
                activation_reason="Importance is too broad.",
                query_role="downweighted",
                should_use_in_provider_query=False,
            ),
            IntentConcept(
                term="possible application",
                category="application",
                source="assumption",
                confidence=0.45,
                activation_reason="Only an assumption.",
                query_role="uncertain",
                should_use_in_provider_query=False,
            ),
        ],
        ignored_or_downweighted_terms=["importance"],
        assumptions=["Importance may mean background or evidence papers."],
    )

    contract = SearchContractAgent().build(
        brief.original_question,
        search_brief=brief,
        expert_intent=intent,
    )

    assert contract.must_include_concepts == ["target effect"]
    assert "Studies discussing why the effect is important" not in contract.must_include_concepts
    assert "surface-sensitive evidence" not in contract.must_include_concepts
    assert contract.optional_concepts == ["candidate probe"]
    assert contract.uncertain_concepts == ["possible application"]
    assert "importance" in contract.dropped_downweighted_terms
    assert contract.constraint_groups
    assert contract.constraint_groups[0].operator == "OR"


def test_cross_domain_cases_keep_domain_concepts_separate():
    cases = [
        json.loads((FIXTURE_DIR / "spaldin_surface_magnetization_case.json").read_text()),
        json.loads((FIXTURE_DIR / "ferroelectric_surface_polarization_case.json").read_text()),
        json.loads((FIXTURE_DIR / "ai_literature_screening_case.json").read_text()),
    ]

    for case in cases:
        question = case["user_question"]
        seed_hints = SeedExtractionAgent().extract(question)
        intent = NoviceIntentInterpreter().repair(question, seed_hints=seed_hints)
        terms = {concept.term.lower() for concept in intent.structured_concepts}
        payload = plan_screening_queries(question)
        plan = payload["query_plan"]
        must_text = " ".join(plan.must_terms).lower()
        query_text = " ".join([*plan.openalex_queries, *plan.semantic_scholar_queries]).lower()

        assert question.lower() not in must_text
        assert not any("studies discussing" in term.lower() for term in plan.must_terms)
        for concept in case.get("expected_concepts", []):
            expected = concept.lower()
            assert (
                any(expected in term for term in terms)
                or expected in query_text
            )
        for concept in case.get("excluded_concepts", []):
            assert concept.lower() not in terms
            assert concept.lower() not in query_text
