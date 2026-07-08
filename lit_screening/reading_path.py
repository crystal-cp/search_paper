"""Generate a recommended reading path from screened, ranked papers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lit_screening.models import RankedPaper
from lit_screening.utils import ensure_dir


def generate_reading_path(
    path: str | Path,
    ranked_papers: list[RankedPaper],
    result_groups: dict[str, list[dict]],
) -> dict[str, int]:
    """Write a Markdown reading path organized by study purpose.

    The reading path is a recommendation layer, not a second screening system.
    It must respect the already-written screening decision, domain decision, and
    reading priority. Excluded or out-of-scope papers are never recommended just
    because a role/lane classifier placed them in a group.
    """

    ranked_by_id = {item.paper.paper_id: item for item in ranked_papers}
    used: set[str] = set()
    sections: list[tuple[str, list[dict[str, Any]]]] = []

    core_rows = _select_ranked(
        ranked_papers,
        allowed_priorities={"must_read"},
        used=used,
        limit=5,
    )
    sections.append(("Core First Reads", core_rows))

    method_rows = _select_grouped(
        result_groups,
        ["implementation_relevant", "background_or_survey"],
        ranked_by_id,
        allowed_priorities={"must_read", "read_later"},
        used=used,
        limit=3,
    )
    sections.append(("Method / Mechanism Papers", method_rows))

    evaluation_rows = _select_grouped(
        result_groups,
        ["evaluation_relevant"],
        ranked_by_id,
        allowed_priorities={"must_read", "read_later"},
        used=used,
        limit=3,
    )
    sections.append(("Evaluation Papers", evaluation_rows))

    frontier_rows = _select_grouped(
        result_groups,
        ["recent_frontier"],
        ranked_by_id,
        allowed_priorities={"read_later"},
        used=used,
        limit=3,
    )
    if not frontier_rows:
        frontier_rows = _select_ranked(
            ranked_papers,
            allowed_priorities={"read_later"},
            used=used,
            limit=3,
        )
    sections.append(("Recent Frontier / Read Later", frontier_rows))

    optional_rows = _select_grouped(
        result_groups,
        ["peripheral"],
        ranked_by_id,
        allowed_priorities={"optional"},
        used=used,
        limit=3,
    )
    if not optional_rows:
        optional_rows = _select_ranked(
            ranked_papers,
            allowed_priorities={"optional"},
            used=used,
            limit=3,
        )
    sections.append(("Optional Peripheral / Background", optional_rows))

    selected_rows = [row for _title, rows in sections for row in rows]
    diagnostics = reading_path_diagnostics(selected_rows, ranked_by_id)

    lines = ["# Recommended Reading Path", ""]
    for title, rows in sections:
        lines.extend([f"## {title}", ""])
        if not rows:
            lines.append("- No suitable papers found in this run.")
        else:
            for row in rows:
                priority = row.get("reading_priority", "")
                lines.append(
                    f"- #{row.get('rank', '')}: {row.get('title', '')}"
                    + (f" ({priority})" if priority else "")
                )
        lines.append("")
    destination = Path(path)
    ensure_dir(destination.parent)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return diagnostics


def reading_path_diagnostics(
    rows: list[dict[str, Any]],
    ranked_by_id: dict[str, RankedPaper],
) -> dict[str, int]:
    """Return recommendation-integrity diagnostics for evaluation.json."""

    paper_ids = [str(row.get("paper_id", "")) for row in rows if row.get("paper_id")]
    exclude_count = 0
    out_of_scope_count = 0
    negative_context_count = 0
    for paper_id in paper_ids:
        item = ranked_by_id.get(paper_id)
        if item is None:
            continue
        decision = item.screening_decision
        domain = item.domain_assessment
        if decision and (
            decision.decision == "exclude"
            or decision.reading_priority == "exclude"
        ):
            exclude_count += 1
        if domain and domain.domain_decision == "out_of_scope":
            out_of_scope_count += 1
        if domain and getattr(domain, "negative_context_match", None):
            negative_context_count += 1
    return {
        "reading_path_paper_count": len(paper_ids),
        "reading_path_exclude_count": exclude_count,
        "reading_path_out_of_scope_count": out_of_scope_count,
        "reading_path_duplicate_count": len(paper_ids) - len(set(paper_ids)),
        "reading_path_negative_context_count": negative_context_count,
    }


def _select_ranked(
    ranked_papers: list[RankedPaper],
    *,
    allowed_priorities: set[str],
    used: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in ranked_papers:
        if len(rows) >= limit:
            break
        if not _is_recommendable(item, allowed_priorities):
            continue
        if item.paper.paper_id in used:
            continue
        rows.append(_row_for_item(item))
        used.add(item.paper.paper_id)
    return rows


def _select_grouped(
    groups: dict[str, list[dict]],
    names: list[str],
    ranked_by_id: dict[str, RankedPaper],
    *,
    allowed_priorities: set[str],
    used: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in names:
        for row in groups.get(name, []):
            if len(rows) >= limit:
                return rows
            paper_id = str(row.get("paper_id", ""))
            item = ranked_by_id.get(paper_id)
            if item is None or not _is_recommendable(item, allowed_priorities):
                continue
            if paper_id in used:
                continue
            rows.append(_row_for_item(item))
            used.add(paper_id)
    return rows


def _is_recommendable(
    item: RankedPaper,
    allowed_priorities: set[str],
) -> bool:
    decision = item.screening_decision
    domain = item.domain_assessment
    if decision is None:
        return False
    if decision.decision == "exclude" or decision.reading_priority == "exclude":
        return False
    if decision.reading_priority not in allowed_priorities:
        return False
    if domain and domain.domain_decision == "out_of_scope":
        return False
    return True


def _row_for_item(item: RankedPaper) -> dict[str, Any]:
    decision = item.screening_decision
    domain = item.domain_assessment
    return {
        "rank": item.rank,
        "title": item.paper.title,
        "paper_id": item.paper.paper_id,
        "decision": decision.decision if decision else "",
        "reading_priority": decision.reading_priority if decision else "",
        "domain_decision": domain.domain_decision if domain else "",
    }
