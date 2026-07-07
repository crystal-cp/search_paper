"""Lightweight preference learning from human feedback labels."""

from __future__ import annotations

from collections import Counter
from typing import Any

from lit_screening.models import (
    FeedbackRecord,
    PreferenceLearningResult,
    RankedPaper,
    SearchContract,
)
from lit_screening.utils import clamp, tokenize


MIN_TRAINING_LABELS = 4


class PreferenceLearningAgent:
    """Learn relevance preferences from include/exclude feedback."""

    def learn(
        self,
        ranked_papers: list[RankedPaper],
        feedback_records: dict[str, FeedbackRecord],
        search_contract: SearchContract | None = None,
    ) -> PreferenceLearningResult:
        """Return learned preference scores and query-refinement terms."""

        labels = {
            paper_id: record.label
            for paper_id, record in feedback_records.items()
            if record.label in {"include", "exclude"}
        }
        include_count = sum(1 for label in labels.values() if label == "include")
        exclude_count = sum(1 for label in labels.values() if label == "exclude")
        if not labels:
            return PreferenceLearningResult(
                enabled=False,
                note="No include/exclude feedback labels were available.",
            )

        labeled_items = [
            item for item in ranked_papers if item.paper.paper_id in labels
        ]
        if not labeled_items:
            return PreferenceLearningResult(
                enabled=False,
                labeled_paper_count=len(labels),
                include_count=include_count,
                exclude_count=exclude_count,
                note="Feedback labels did not match ranked paper IDs.",
            )

        if (
            len(labeled_items) >= MIN_TRAINING_LABELS
            and include_count > 0
            and exclude_count > 0
        ):
            result = self._learn_with_classifier(
                ranked_papers,
                labeled_items,
                labels,
                search_contract,
            )
        else:
            result = self._learn_with_term_frequency(
                ranked_papers,
                labeled_items,
                labels,
                search_contract,
            )
        result.labeled_paper_count = len(labeled_items)
        result.include_count = include_count
        result.exclude_count = exclude_count
        return result

    def query_refinement_payload(
        self,
        result: PreferenceLearningResult,
    ) -> dict[str, Any]:
        """Return query-plan edits suggested by learned preferences."""

        return {
            "enabled": result.enabled,
            "model_type": result.model_type,
            "positive_terms": result.positive_terms,
            "negative_terms": result.negative_terms,
            "suggested_must_terms": result.suggested_must_terms,
            "suggested_optional_terms": result.suggested_optional_terms,
            "suggested_exclude_terms": result.suggested_exclude_terms,
            "note": result.note,
        }

    def _learn_with_classifier(
        self,
        ranked_papers: list[RankedPaper],
        labeled_items: list[RankedPaper],
        labels: dict[str, str],
        search_contract: SearchContract | None,
    ) -> PreferenceLearningResult:
        """Train a small TF-IDF + Logistic Regression classifier."""

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
        except Exception:
            return self._learn_with_term_frequency(
                ranked_papers,
                labeled_items,
                labels,
                search_contract,
                note="scikit-learn was unavailable; used term-frequency fallback.",
            )

        documents = [paper_text(item) for item in labeled_items]
        y = [1 if labels[item.paper.paper_id] == "include" else 0 for item in labeled_items]
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=500,
                min_df=1,
            )
            x_train = vectorizer.fit_transform(documents)
            classifier = LogisticRegression(max_iter=1000)
            classifier.fit(x_train, y)
            feature_names = list(vectorizer.get_feature_names_out())
            coefficients = classifier.coef_[0]
            positive_terms = top_weighted_terms(feature_names, coefficients, positive=True)
            negative_terms = top_weighted_terms(feature_names, coefficients, positive=False)
            x_all = vectorizer.transform([paper_text(item) for item in ranked_papers])
            probabilities = classifier.predict_proba(x_all)[:, 1]
            scores = {
                item.paper.paper_id: round(float(probability), 4)
                for item, probability in zip(ranked_papers, probabilities)
            }
            return PreferenceLearningResult(
                enabled=True,
                model_type="tfidf_logistic_regression",
                preference_scores=scores,
                positive_terms=positive_terms,
                negative_terms=negative_terms,
                **suggestions_from_terms(
                    positive_terms,
                    negative_terms,
                    search_contract,
                ),
            )
        except Exception:
            return self._learn_with_term_frequency(
                ranked_papers,
                labeled_items,
                labels,
                search_contract,
                note="Classifier training failed; used term-frequency fallback.",
            )

    def _learn_with_term_frequency(
        self,
        ranked_papers: list[RankedPaper],
        labeled_items: list[RankedPaper],
        labels: dict[str, str],
        search_contract: SearchContract | None,
        note: str = "Too few labels for a classifier; used term-frequency fallback.",
    ) -> PreferenceLearningResult:
        """Compare include and exclude term frequencies."""

        positive_counter: Counter[str] = Counter()
        negative_counter: Counter[str] = Counter()
        for item in labeled_items:
            terms = preference_terms(paper_text(item))
            if labels[item.paper.paper_id] == "include":
                positive_counter.update(terms)
            elif labels[item.paper.paper_id] == "exclude":
                negative_counter.update(terms)

        positive_terms = contrastive_terms(positive_counter, negative_counter)
        negative_terms = contrastive_terms(negative_counter, positive_counter)
        scores = {
            item.paper.paper_id: round(
                term_frequency_preference_score(
                    item,
                    positive_terms,
                    negative_terms,
                ),
                4,
            )
            for item in ranked_papers
        }
        return PreferenceLearningResult(
            enabled=True,
            model_type="term_frequency",
            preference_scores=scores,
            positive_terms=positive_terms,
            negative_terms=negative_terms,
            note=note,
            **suggestions_from_terms(positive_terms, negative_terms, search_contract),
        )


def paper_text(item: RankedPaper) -> str:
    """Build text used for preference learning."""

    return " ".join(
        [
            item.paper.title,
            item.paper.abstract,
            item.paper.venue,
            " ".join(item.paper.fields_of_study),
            item.paper.tldr,
            item.evidence.claim,
            item.evidence.evidence_sentence,
            item.evidence.relevance_reason,
        ]
    )


def preference_terms(text: str) -> list[str]:
    """Extract unigrams and bigrams suitable for preference summaries."""

    tokens = tokenize(text)
    terms = [token for token in tokens if not token.isdigit()]
    bigrams = [
        f"{left} {right}"
        for left, right in zip(tokens, tokens[1:])
        if not left.isdigit() and not right.isdigit()
    ]
    return terms + bigrams


def contrastive_terms(
    primary: Counter[str],
    other: Counter[str],
    limit: int = 12,
) -> list[str]:
    """Return terms that are more common in one label group than the other."""

    scored: list[tuple[float, str]] = []
    for term, count in primary.items():
        if len(term) < 3:
            continue
        score = count - other.get(term, 0)
        if score <= 0:
            continue
        scored.append((score + 0.01 * count, term))
    scored.sort(reverse=True)
    return [term for _, term in scored[:limit]]


def term_frequency_preference_score(
    item: RankedPaper,
    positive_terms: list[str],
    negative_terms: list[str],
) -> float:
    """Score a paper from learned positive and negative terms."""

    text = paper_text(item).lower()
    positive_hits = sum(1 for term in positive_terms if term.lower() in text)
    negative_hits = sum(1 for term in negative_terms if term.lower() in text)
    total_terms = max(1, len(positive_terms) + len(negative_terms))
    centered = (positive_hits - negative_hits) / total_terms
    return clamp(0.5 + centered)


def top_weighted_terms(
    feature_names: list[str],
    coefficients: Any,
    positive: bool,
    limit: int = 12,
) -> list[str]:
    """Return strongest positive or negative classifier features."""

    pairs = list(zip(feature_names, coefficients))
    pairs.sort(key=lambda item: item[1], reverse=positive)
    terms: list[str] = []
    for term, weight in pairs:
        if positive and weight <= 0:
            continue
        if not positive and weight >= 0:
            continue
        cleaned = " ".join(str(term).split())
        if cleaned and cleaned not in terms:
            terms.append(cleaned)
        if len(terms) >= limit:
            break
    return terms


def suggestions_from_terms(
    positive_terms: list[str],
    negative_terms: list[str],
    search_contract: SearchContract | None,
) -> dict[str, list[str]]:
    """Build query refinement suggestions from preference terms."""

    must_terms = search_contract.must_include_concepts if search_contract else []
    existing_must = set(must_terms)
    existing_exclude = set(search_contract.must_exclude_concepts if search_contract else [])
    positive_context = " ".join([*must_terms, *positive_terms[:8]]).lower()
    suggested_must = [
        term for term in positive_terms[:5] if term not in existing_must
    ]
    suggested_optional = [
        term for term in positive_terms[5:10] if term not in existing_must
    ]
    suggested_exclude = [
        term
        for term in negative_terms[:8]
        if term not in existing_exclude
        and not term_conflicts_with_positive_context(term, positive_context)
    ]
    return {
        "suggested_must_terms": unique_terms(suggested_must),
        "suggested_optional_terms": unique_terms(suggested_optional),
        "suggested_exclude_terms": unique_terms(suggested_exclude),
    }


def term_conflicts_with_positive_context(term: str, positive_context: str) -> bool:
    """Return True when an exclude term would remove a positive concept."""

    lowered = term.lower()
    if not lowered:
        return True
    if lowered in positive_context:
        return True
    return False


def unique_terms(terms: list[str]) -> list[str]:
    """Return unique readable terms."""

    result: list[str] = []
    for term in terms:
        cleaned = " ".join(term.split())
        if len(cleaned) < 3:
            continue
        if cleaned not in result:
            result.append(cleaned)
    return result
