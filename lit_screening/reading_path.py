"""Generate a recommended reading path from ranked papers."""

from __future__ import annotations

from pathlib import Path

from lit_screening.models import RankedPaper
from lit_screening.utils import ensure_dir


def generate_reading_path(
    path: str | Path,
    ranked_papers: list[RankedPaper],
    result_groups: dict[str, list[dict]],
) -> None:
    """Write a Markdown reading path organized by study purpose."""

    lines = ["# Recommended Reading Path", ""]
    sections = [
        ("Overview / Background", _select(result_groups, ["background_or_survey", "must_read"], 2)),
        ("Core System / Method Papers", _select(result_groups, ["implementation_relevant", "must_read"], 3)),
        ("Recent Frontier Papers", _select(result_groups, ["recent_frontier", "must_read"], 2)),
        ("Evaluation Papers", _select(result_groups, ["evaluation_relevant", "must_read"], 2)),
        ("Optional Peripheral Papers", _select(result_groups, ["peripheral"], 3)),
    ]
    fallback = [
        {"rank": item.rank, "title": item.paper.title, "paper_id": item.paper.paper_id}
        for item in ranked_papers
    ]
    for title, rows in sections:
        if not rows and fallback:
            rows = fallback[:2]
        lines.extend([f"## {title}", ""])
        if not rows:
            lines.append("- No suitable papers found in this run.")
        else:
            for row in rows:
                lines.append(f"- #{row.get('rank', '')}: {row.get('title', '')}")
        lines.append("")
    destination = Path(path)
    ensure_dir(destination.parent)
    destination.write_text("\n".join(lines), encoding="utf-8")


def _select(
    groups: dict[str, list[dict]],
    names: list[str],
    limit: int,
) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for name in names:
        for row in groups.get(name, []):
            paper_id = row.get("paper_id", "")
            if paper_id in seen:
                continue
            rows.append(row)
            seen.add(paper_id)
            if len(rows) >= limit:
                return rows
    return rows
