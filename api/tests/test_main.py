import json

import db
import inference
import main

MINIMAL_TRANSACTION_ROW = {
    "transaction_id": "T1", "source_system": "core_transactions", "account_id": "ACCT-1",
    "event_ts": "2026-01-01T10:00:00", "event_date": "2026-01-01", "amount_usd": 10.0,
}


class TestHealthAndMeta:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_meta_returns_reference_lists(self, client):
        resp = client.get("/meta")
        body = resp.json()
        assert "Groceries" in body["categories"]
        assert "online" in body["channels"]
        assert "London" in body["cities"]


class TestTransactions:
    def test_list_transactions_paginates_and_passes_filters(self, client, monkeypatch):
        captured = {}

        def fake_fetch(**kwargs):
            captured.update(kwargs)
            return [dict(MINIMAL_TRANSACTION_ROW)], 1

        monkeypatch.setattr(db, "fetch_transactions", fake_fetch)
        resp = client.get("/transactions", params={"limit": 10, "category": "Fuel", "predicted_label": 1})

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["transaction_id"] == "T1"
        assert captured["category"] == "Fuel"
        assert captured["predicted_label"] == 1
        assert captured["limit"] == 10

    def test_get_transaction_404_when_missing(self, client, monkeypatch):
        monkeypatch.setattr(db, "fetch_transaction_by_id", lambda txn_id: None)
        resp = client.get("/transactions/DOES-NOT-EXIST")
        assert resp.status_code == 404

    def test_get_transaction_found(self, client, monkeypatch):
        monkeypatch.setattr(db, "fetch_transaction_by_id", lambda txn_id: dict(MINIMAL_TRANSACTION_ROW))
        resp = client.get("/transactions/T1")
        assert resp.status_code == 200
        assert resp.json()["transaction_id"] == "T1"


class TestStats:
    def test_summary_404_when_no_data_yet(self, client, monkeypatch):
        monkeypatch.setattr(db, "fetch_summary_stats", lambda: {"total_transactions": 0})
        resp = client.get("/stats/summary")
        assert resp.status_code == 404

    def test_summary_includes_model_metrics_when_present(self, client, monkeypatch, tmp_path):
        stats = {
            "total_transactions": 100, "actual_fraud_count": 2, "predicted_fraud_count": 3,
            "total_amount_usd": 1000.0, "flagged_amount_usd": 50.0, "avg_fraud_probability": 0.1,
            "earliest_date": "2026-01-01", "latest_date": "2026-01-31",
        }
        monkeypatch.setattr(db, "fetch_summary_stats", lambda: dict(stats))

        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps({"auc_roc": 0.78}))
        monkeypatch.setattr(main, "METRICS_PATH", str(metrics_file))

        resp = client.get("/stats/summary")
        assert resp.status_code == 200
        assert resp.json()["model_metrics"] == {"auc_roc": 0.78}

    def test_timeseries_and_categories_pass_through(self, client, monkeypatch):
        monkeypatch.setattr(db, "fetch_timeseries", lambda: [])
        monkeypatch.setattr(db, "fetch_category_breakdown", lambda: [])
        assert client.get("/stats/timeseries").json() == []
        assert client.get("/stats/categories").json() == []


class TestDataQualityReport:
    def test_404_when_report_missing(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(main, "DQ_REPORT_PATH", str(tmp_path / "nope.json"))
        resp = client.get("/reports/data-quality")
        assert resp.status_code == 404

    def test_returns_report_contents_when_present(self, client, monkeypatch, tmp_path):
        report_file = tmp_path / "dq.json"
        report_file.write_text(json.dumps({"duplicates_removed": 42}))
        monkeypatch.setattr(main, "DQ_REPORT_PATH", str(report_file))
        resp = client.get("/reports/data-quality")
        assert resp.status_code == 200
        assert resp.json() == {"duplicates_removed": 42}


class TestScore:
    def test_rejects_unknown_category(self, client):
        resp = client.post("/score", json={"amount_usd": 10.0, "category": "NotACategory", "channel": "online"})
        assert resp.status_code == 422

    def test_rejects_unknown_channel(self, client):
        resp = client.post("/score", json={"amount_usd": 10.0, "category": "Fuel", "channel": "carrier-pigeon"})
        assert resp.status_code == 422

    def test_rejects_non_positive_amount(self, client):
        resp = client.post("/score", json={"amount_usd": 0, "category": "Fuel", "channel": "online"})
        assert resp.status_code == 422

    def test_successful_score_delegates_to_inference(self, client, monkeypatch):
        canned = {
            "fraud_probability": 0.42, "predicted_label": 0, "rule_flags": [],
            "used_account_history": False, "features_used": {},
        }
        monkeypatch.setattr(inference, "score", lambda req: canned)
        resp = client.post("/score", json={"amount_usd": 10.0, "category": "Fuel", "channel": "online"})
        assert resp.status_code == 200
        assert resp.json() == canned

    def test_scoring_failure_returns_503_not_a_crash(self, client, monkeypatch):
        def boom(req):
            raise RuntimeError("model not loaded")

        monkeypatch.setattr(inference, "score", boom)
        resp = client.post("/score", json={"amount_usd": 10.0, "category": "Fuel", "channel": "online"})
        assert resp.status_code == 503
