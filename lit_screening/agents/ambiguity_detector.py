"""Ambiguity detection before query planning."""

from __future__ import annotations

from typing import Any


class AmbiguityDetectorAgent:
    """Detect common ambiguous search terms and propose contract constraints."""

    def analyze(self, question: str) -> list[dict[str, Any]]:
        """Return ambiguity records for terms found in the question."""

        text = " ".join(question.split())
        lowered = text.lower()
        records: list[dict[str, Any]] = []
        if "screening" in lowered:
            records.append(self._screening_record(lowered))
        if "agent" in lowered or "multi-agent" in lowered:
            records.append(self._agent_record(lowered))
        if "evidence" in lowered:
            records.append(self._evidence_record(lowered))
        if "optimization" in lowered:
            records.append(self._optimization_record(lowered))
        if "ranking" in lowered:
            records.append(self._ranking_record(lowered))
        return records

    def _screening_record(self, lowered: str) -> dict[str, Any]:
        possible = [
            "literature screening",
            "patient screening",
            "drug screening",
            "biomarker screening",
            "high-throughput materials screening",
        ]
        if "literature screening" in lowered or "abstract screening" in lowered:
            selected = "literature screening"
            must = ["literature screening", "scientific literature", "abstract screening"]
            exclude = [
                "patient screening",
                "drug screening",
                "biomarker screening",
                "high-throughput materials screening",
            ]
        elif "materials screening" in lowered or "high-throughput" in lowered:
            selected = "high-throughput materials screening"
            must = ["materials screening", "high-throughput screening"]
            exclude = ["patient screening", "drug screening", "biomarker screening"]
        elif any(term in lowered for term in ["patient", "clinical", "disease"]):
            selected = "patient screening"
            must = ["patient screening", "clinical screening"]
            exclude = ["literature screening", "materials screening"]
        elif any(term in lowered for term in ["drug", "compound", "biomarker"]):
            selected = "drug or biomarker screening"
            must = ["drug screening", "biomarker screening"]
            exclude = ["literature screening", "materials screening"]
        else:
            selected = "unspecified screening"
            must = []
            exclude = []
        return {
            "term": "screening",
            "possible_meanings": possible,
            "selected_meaning": selected,
            "recommended_must_terms": must,
            "recommended_exclude_terms": exclude,
            "explanation": "The word screening is domain-dependent, so retrieval should bind it to the intended domain.",
        }

    def _agent_record(self, lowered: str) -> dict[str, Any]:
        possible = [
            "software/LLM agent",
            "biological agent",
            "chemical agent",
            "infectious agent",
            "human agent",
        ]
        if any(term in lowered for term in ["llm agent", "llm agents", "multi-agent", "software agent"]):
            selected = "software/LLM agent"
            must = ["LLM agent", "software agent"]
            exclude = ["biological agent", "chemical agent", "infectious agent"]
        elif any(term in lowered for term in ["infectious", "pathogen", "biological"]):
            selected = "biological or infectious agent"
            must = ["biological agent"]
            exclude = ["LLM agent", "software agent"]
        elif "chemical" in lowered:
            selected = "chemical agent"
            must = ["chemical agent"]
            exclude = ["LLM agent", "software agent"]
        else:
            selected = "unspecified agent"
            must = []
            exclude = ["biological agent", "chemical agent", "infectious agent"]
        return {
            "term": "agent",
            "possible_meanings": possible,
            "selected_meaning": selected,
            "recommended_must_terms": must,
            "recommended_exclude_terms": exclude,
            "explanation": "The word agent can refer to software, organisms, chemicals, or people.",
        }

    def _evidence_record(self, lowered: str) -> dict[str, Any]:
        possible = [
            "scientific evidence",
            "evidence verification",
            "legal evidence",
            "evidence-based medicine",
        ]
        if any(term in lowered for term in ["verification", "claim", "abstract", "literature"]):
            selected = "evidence verification in scientific literature"
            must = ["evidence verification", "scientific evidence"]
            exclude = ["legal evidence", "evidence-based medicine"]
        else:
            selected = "scientific evidence"
            must = ["scientific evidence"]
            exclude = ["legal evidence"]
        return {
            "term": "evidence",
            "possible_meanings": possible,
            "selected_meaning": selected,
            "recommended_must_terms": must,
            "recommended_exclude_terms": exclude,
            "explanation": "Evidence should be tied to the research evidence chain rather than unrelated legal or clinical uses.",
        }

    def _optimization_record(self, lowered: str) -> dict[str, Any]:
        possible = [
            "mathematical optimization",
            "structure optimization",
            "hyperparameter optimization",
            "process optimization",
        ]
        if any(term in lowered for term in ["structure", "crystal", "vasp", "relaxation"]):
            selected = "structure optimization"
            must = ["structure optimization"]
            exclude = ["hyperparameter optimization"]
        elif any(term in lowered for term in ["model", "training", "hyperparameter"]):
            selected = "hyperparameter optimization"
            must = ["hyperparameter optimization"]
            exclude = ["structure optimization"]
        else:
            selected = "mathematical optimization"
            must = ["optimization"]
            exclude = []
        return {
            "term": "optimization",
            "possible_meanings": possible,
            "selected_meaning": selected,
            "recommended_must_terms": must,
            "recommended_exclude_terms": exclude,
            "explanation": "Optimization has different meanings in materials, machine learning, and operations research.",
        }

    def _ranking_record(self, lowered: str) -> dict[str, Any]:
        possible = [
            "information retrieval ranking",
            "statistical ranking",
            "sports or league ranking",
            "relevance ranking",
        ]
        if any(term in lowered for term in ["retrieval", "search", "literature", "paper", "relevance"]):
            selected = "information retrieval ranking"
            must = ["relevance ranking", "information retrieval"]
            exclude = ["sports ranking"]
        else:
            selected = "relevance ranking"
            must = ["ranking"]
            exclude = ["sports ranking"]
        return {
            "term": "ranking",
            "possible_meanings": possible,
            "selected_meaning": selected,
            "recommended_must_terms": must,
            "recommended_exclude_terms": exclude,
            "explanation": "Ranking should describe relevance ordering unless the user asks for another ranking domain.",
        }
