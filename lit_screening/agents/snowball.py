"""Optional citation snowballing scaffold.

The MVP keeps this disabled by default so normal runs do not spend extra API
requests. It is structured so a future run can enable Semantic Scholar citation,
reference, or recommendation expansion around top seed papers.
"""

from __future__ import annotations

from dataclasses import dataclass

from lit_screening.models import Paper, RankedPaper


@dataclass
class RetrievalPath:
    """Trace how an expanded paper entered the candidate set."""

    paper_id: str
    source_stage: str
    seed_paper_id: str
    seed_title: str
    reason: str


class CitationSnowballAgent:
    """Optional citation-expansion agent, disabled by default."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def expand(self, ranked_papers: list[RankedPaper], top_n: int = 3) -> tuple[list[Paper], list[RetrievalPath]]:
        """Return citation-expanded papers.

        The current MVP intentionally returns no papers unless a future
        Semantic Scholar implementation enables network expansion.
        """

        if not self.enabled:
            return [], []
        seeds = ranked_papers[:top_n]
        paths = [
            RetrievalPath(
                paper_id=seed.paper.paper_id,
                source_stage="seed",
                seed_paper_id=seed.paper.paper_id,
                seed_title=seed.paper.title,
                reason="Top-ranked seed paper selected for future citation expansion.",
            )
            for seed in seeds
        ]
        return [], paths
