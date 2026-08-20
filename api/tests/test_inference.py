from datetime import datetime, timezone

import db
import inference
from schemas import ScoreRequest


def _req(**overrides):
    defaults = {"amount_usd": 50.0, "category": "Groceries", "channel": "online"}
    defaults.update(overrides)
    return ScoreRequest(**defaults)


class TestBuildFeaturesBrandNewCard:
    def test_no_account_id_means_neutral_history(self):
        features, used_history, rule_flags = inference.build_features(_req())

        assert used_history is False
        assert features["txn_count_1h"] == 1  # just this transaction, no prior history
        assert features["txn_amount_sum_1h"] == 50.0
        assert features["amount_zscore"] == 0.0
        assert features["seconds_since_last_txn"] == -1.0
        assert features["distance_from_last_txn_km"] == -1.0
        assert features["is_new_merchant"] == 1
        assert "new_merchant" in rule_flags


class TestBuildFeaturesExistingAccount:
    def test_falls_back_to_new_card_when_account_has_no_history(self, monkeypatch):
        # account_id given, but the account has never transacted before
        monkeypatch.setattr(db, "fetch_account_recent_activity", lambda *a, **k: ({"hist_txn_count": 0}, None))
        features, used_history, _ = inference.build_features(_req(account_id="ACCT-NEW"))
        assert used_history is False
        assert features["txn_count_1h"] == 1

    def test_uses_real_velocity_and_zscore_when_history_exists(self, monkeypatch):
        stats = {
            "hist_txn_count": 10, "txn_count_1h": 2, "txn_count_24h": 5,
            "txn_amount_sum_1h": 40.0, "txn_amount_sum_24h": 200.0,
            "hist_avg_amount": 20.0, "hist_stddev_amount": 5.0,
        }
        monkeypatch.setattr(db, "fetch_account_recent_activity", lambda *a, **k: (stats, None))
        monkeypatch.setattr(db, "account_has_category", lambda *a, **k: True)

        features, used_history, rule_flags = inference.build_features(_req(amount_usd=100.0, account_id="ACCT-1"))

        assert used_history is True
        # this transaction adds itself to the trailing windows on top of history
        assert features["txn_count_1h"] == 3
        assert features["txn_amount_sum_1h"] == 140.0
        # (100 - 20) / 5 = 16 -> a large outlier
        assert features["amount_zscore"] == 16.0
        assert features["rule_flag_amount_outlier"] == 1
        assert "amount_outlier" in rule_flags
        assert features["is_new_merchant"] == 0  # account_has_category returned True
        assert "new_merchant" not in rule_flags

    def test_zero_stddev_history_keeps_zscore_neutral(self, monkeypatch):
        # a brand-new-ish account with only one prior transaction has stddev=0/None -
        # must not divide by zero
        stats = {
            "hist_txn_count": 1, "txn_count_1h": 1, "txn_count_24h": 1,
            "txn_amount_sum_1h": 20.0, "txn_amount_sum_24h": 20.0,
            "hist_avg_amount": 20.0, "hist_stddev_amount": None,
        }
        monkeypatch.setattr(db, "fetch_account_recent_activity", lambda *a, **k: (stats, None))
        monkeypatch.setattr(db, "account_has_category", lambda *a, **k: True)

        features, _, _ = inference.build_features(_req(account_id="ACCT-1"))
        assert features["amount_zscore"] == 0.0

    def test_impossible_travel_detected_for_far_fast_transaction(self, monkeypatch):
        stats = {
            "hist_txn_count": 5, "txn_count_1h": 1, "txn_count_24h": 1,
            "txn_amount_sum_1h": 10.0, "txn_amount_sum_24h": 10.0,
            "hist_avg_amount": 10.0, "hist_stddev_amount": 2.0,
        }
        last_txn = {"event_ts": datetime(2026, 1, 1, 10, 0, 0), "city": "New York", "category": "Groceries"}
        monkeypatch.setattr(db, "fetch_account_recent_activity", lambda *a, **k: (stats, last_txn))
        monkeypatch.setattr(db, "account_has_category", lambda *a, **k: True)

        # 10 minutes later in London - physically impossible
        req = _req(account_id="ACCT-1", city="London", event_ts=datetime(2026, 1, 1, 10, 10, 0, tzinfo=timezone.utc))
        features, _, rule_flags = inference.build_features(req)

        assert features["distance_from_last_txn_km"] > 0
        assert features["impossible_travel_flag"] == 1
        assert "impossible_travel" in rule_flags

    def test_no_distance_when_either_city_unknown(self, monkeypatch):
        stats = {
            "hist_txn_count": 5, "txn_count_1h": 1, "txn_count_24h": 1,
            "txn_amount_sum_1h": 10.0, "txn_amount_sum_24h": 10.0,
            "hist_avg_amount": 10.0, "hist_stddev_amount": 2.0,
        }
        last_txn = {"event_ts": datetime(2026, 1, 1, 10, 0, 0), "city": "Nowhereville", "category": "Groceries"}
        monkeypatch.setattr(db, "fetch_account_recent_activity", lambda *a, **k: (stats, last_txn))
        monkeypatch.setattr(db, "account_has_category", lambda *a, **k: True)

        req = _req(account_id="ACCT-1", city="London", event_ts=datetime(2026, 1, 1, 10, 10, 0, tzinfo=timezone.utc))
        features, _, _ = inference.build_features(req)
        assert features["distance_from_last_txn_km"] == -1.0
        assert features["impossible_travel_flag"] == 0


class TestOddHourAndDayOfWeek:
    def test_is_odd_hour_flag(self):
        features, _, rule_flags = inference.build_features(_req(event_ts=datetime(2026, 1, 5, 2, 30, tzinfo=timezone.utc)))
        assert features["is_odd_hour"] == 1
        assert "odd_hour" in rule_flags

    def test_not_odd_hour_at_noon(self):
        features, _, rule_flags = inference.build_features(_req(event_ts=datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)))
        assert features["is_odd_hour"] == 0
        assert "odd_hour" not in rule_flags

    def test_day_of_week_matches_spark_convention(self):
        # 2026-01-04 is a Sunday; Spark's dayofweek() defines Sunday=1
        features, _, _ = inference.build_features(_req(event_ts=datetime(2026, 1, 4, 12, 0, tzinfo=timezone.utc)))
        assert features["day_of_week"] == 1
        # 2026-01-05 is a Monday -> Spark's dayofweek()=2
        features, _, _ = inference.build_features(_req(event_ts=datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)))
        assert features["day_of_week"] == 2


class TestScore:
    def test_formats_model_output_into_response_dict(self, monkeypatch):
        fake_prediction = {"probability": [0.13, 0.87], "prediction": 1}

        class FakeDF:
            def select(self, *a, **k):
                return self

            def first(self):
                return fake_prediction

        class FakeModel:
            def transform(self, df):
                return FakeDF()

        class FakeSpark:
            def createDataFrame(self, rows):
                return object()

        monkeypatch.setattr(inference, "_get_model", lambda: (FakeSpark(), FakeModel()))

        result = inference.score(_req())
        assert result["fraud_probability"] == 0.87
        assert result["predicted_label"] == 1
        assert result["used_account_history"] is False
        assert "features_used" in result
