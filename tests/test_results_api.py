from fastapi.testclient import TestClient
import pytest

from src.pipeline import api


@pytest.mark.parametrize("endpoint", ["/results/summary", "/results/sensitivity"])
def test_missing_results_are_reported_as_unavailable(endpoint, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DOCS_DIR", tmp_path)
    response = TestClient(api.app).get(endpoint)
    assert response.status_code == 503


def test_summary_reads_committed_evaluation_artifacts():
    response = TestClient(api.app).get("/results/summary")
    assert response.status_code == 200
    result = response.json()
    assert "clean" in result["confusion_matrices"]
    rows = {row["condition"]: row for row in result["degradation_benchmarks"]}
    assert "Clean Baseline" in rows
    assert "GSM Codec" in rows
    assert "±" in rows["Clean Baseline"]["accuracy_ci"]


def test_sensitivity_reads_committed_evaluation_artifact():
    response = TestClient(api.app).get("/results/sensitivity")
    assert response.status_code == 200
    assert len(response.json()) > 0
