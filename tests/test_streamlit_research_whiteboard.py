import json
from pathlib import Path

import app


def test_research_whiteboard_helpers_import_and_read_list_artifacts(tmp_path):
    artifact_path = tmp_path / "paper_roles.json"
    artifact_path.write_text(
        json.dumps(
            [
                {
                    "paper_id": "p1",
                    "roles": ["theory_origin", "conceptual_framework"],
                    "primary_role": "theory_origin",
                },
                {
                    "paper_id": "p2",
                    "roles": ["theory_origin"],
                    "primary_role": "theory_origin",
                },
            ]
        ),
        encoding="utf-8",
    )

    payload = app.read_json_artifact(artifact_path)
    records = app.as_records(payload)
    distribution = app.count_values(records, "roles")

    assert hasattr(app, "render_research_whiteboard")
    assert len(records) == 2
    assert distribution.to_dict("records")[0] == {
        "value": "theory_origin",
        "count": 2,
    }
    assert app.read_json_artifact(tmp_path / "missing.json") is None
    assert app.artifact_missing_message("English") == "Not available for this run."
    assert "Research Whiteboard" in Path("app.py").read_text(encoding="utf-8")
