"""Load lightweight domain-pack JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lit_screening.models import DomainConcept, DomainPack


PACK_DIR = Path(__file__).resolve().parent


def list_domain_packs() -> list[str]:
    """Return available domain-pack names."""

    return sorted(path.stem for path in PACK_DIR.glob("*.json") if path.is_file())


def load_domain_pack(domain_name: str) -> DomainPack:
    """Load a domain pack by name.

    Raises:
        ValueError: If the requested domain pack does not exist.
    """

    cleaned_name = domain_name.strip()
    path = PACK_DIR / f"{cleaned_name}.json"
    if not cleaned_name or not path.exists():
        available = ", ".join(list_domain_packs()) or "none"
        raise ValueError(
            f"Domain pack '{domain_name}' does not exist. Available domain packs: {available}."
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    return _domain_pack_from_json(data)


def _domain_pack_from_json(data: dict[str, Any]) -> DomainPack:
    """Convert raw JSON data into a DomainPack dataclass."""

    concepts = {
        name: DomainConcept(
            synonyms=_string_list(payload.get("synonyms", [])),
            related=_string_list(payload.get("related", [])),
        )
        for name, payload in dict(data.get("concepts", {})).items()
        if isinstance(payload, dict)
    }
    return DomainPack(
        domain_name=str(data.get("domain_name") or ""),
        concepts=concepts,
        materials=_string_list(data.get("materials", [])),
        methods=_string_list(data.get("methods", [])),
        applications=_string_list(data.get("applications", [])),
        false_positive_terms=_string_list(data.get("false_positive_terms", [])),
    )


def _string_list(value: Any) -> list[str]:
    """Return a compact list of string values."""

    if not isinstance(value, list):
        return []
    return [" ".join(item.split()) for item in value if isinstance(item, str) and item.strip()]
