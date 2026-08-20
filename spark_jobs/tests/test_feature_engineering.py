from datetime import datetime

import feature_engineering as fe


class TestHaversineKm:
    def test_same_point_is_zero_distance(self, spark):
        df = spark.createDataFrame([(40.7128, -74.0060, 40.7128, -74.0060)], ["lat1", "lon1", "lat2", "lon2"])
        row = df.withColumn(
            "d", fe.haversine_km(df.lat1, df.lon1, df.lat2, df.lon2)
        ).collect()[0]
        assert row["d"] == 0.0

    def test_new_york_to_london_is_roughly_right(self, spark):
        # well-known great-circle distance, ~5570 km
        df = spark.createDataFrame([(40.7128, -74.0060, 51.5074, -0.1278)], ["lat1", "lon1", "lat2", "lon2"])
        row = df.withColumn("d", fe.haversine_km(df.lat1, df.lon1, df.lat2, df.lon2)).collect()[0]
        assert 5400 < row["d"] < 5700


class TestEngineerFeatures:
    def _base_rows(self):
        # two transactions on the same account an hour apart in the same
        # city, plus a third on a different account entirely
        return [
            ("T1", "ACCT-1", datetime(2026, 1, 1, 10, 0, 0), 50.0, "Groceries", "online",
             "New York", 40.7128, -74.0060, "M1", 0, "2026-01-01"),
            ("T2", "ACCT-1", datetime(2026, 1, 1, 10, 30, 0), 60.0, "Groceries", "online",
             "New York", 40.7128, -74.0060, "M1", 0, "2026-01-01"),
            ("T3", "ACCT-2", datetime(2026, 1, 1, 10, 0, 0), 20.0, "Fuel", "pos",
             "London", 51.5074, -0.1278, "M2", 0, "2026-01-01"),
        ]

    def _df(self, spark, rows):
        return spark.createDataFrame(
            rows,
            ["transaction_id", "account_id", "event_ts", "amount_usd", "category", "channel",
             "city", "lat", "lon", "merchant", "is_fraud_label", "event_date"],
        )

    def test_velocity_counts_include_current_transaction(self, spark):
        df = self._df(spark, self._base_rows())
        out = {r["transaction_id"]: r for r in fe.engineer_features(df).collect()}

        # T1 is the first transaction for ACCT-1: count-so-far (including itself) is 1
        assert out["T1"]["txn_count_1h"] == 1
        # T2 is 30 minutes after T1 on the same account: falls inside the trailing 1h window
        assert out["T2"]["txn_count_1h"] == 2
        assert out["T2"]["txn_amount_sum_1h"] == 110.0
        # ACCT-2 is unrelated to ACCT-1's history
        assert out["T3"]["txn_count_1h"] == 1

    def test_is_new_merchant_flags_only_the_first_visit(self, spark):
        df = self._df(spark, self._base_rows())
        out = {r["transaction_id"]: r for r in fe.engineer_features(df).collect()}
        assert out["T1"]["is_new_merchant"] == 1
        assert out["T2"]["is_new_merchant"] == 0  # same account, same merchant, second visit
        assert out["T3"]["is_new_merchant"] == 1  # different merchant entirely

    def test_impossible_travel_flag_on_far_apart_fast_transactions(self, spark):
        rows = [
            ("T1", "ACCT-1", datetime(2026, 1, 1, 10, 0, 0), 50.0, "Groceries", "online",
             "New York", 40.7128, -74.0060, "M1", 0, "2026-01-01"),
            # London 10 minutes later - physically impossible for the same card holder
            ("T2", "ACCT-1", datetime(2026, 1, 1, 10, 10, 0), 50.0, "Groceries", "online",
             "London", 51.5074, -0.1278, "M1", 0, "2026-01-01"),
        ]
        df = self._df(spark, rows)
        out = {r["transaction_id"]: r for r in fe.engineer_features(df).collect()}
        assert out["T2"]["impossible_travel_flag"] == 1
        assert "impossible_travel" in out["T2"]["rule_flags"]

    def test_first_transaction_has_neutral_history_features(self, spark):
        df = self._df(spark, self._base_rows())
        out = {r["transaction_id"]: r for r in fe.engineer_features(df).collect()}
        assert out["T1"]["amount_zscore"] == 0.0  # no history yet -> neutral, not an error
        assert out["T1"]["seconds_since_last_txn"] == -1  # fillna sentinel for "no prior transaction"

    def test_rule_flag_high_velocity_at_five_transactions_in_an_hour(self, spark):
        rows = [
            (f"T{i}", "ACCT-1", datetime(2026, 1, 1, 10, i, 0), 10.0, "Groceries", "online",
             "New York", 40.7128, -74.0060, "M1", 0, "2026-01-01")
            for i in range(5)
        ]
        df = self._df(spark, rows)
        out = {r["transaction_id"]: r for r in fe.engineer_features(df).collect()}
        assert out["T4"]["txn_count_1h"] == 5
        assert out["T4"]["rule_flag_high_velocity"] == 1
        assert "high_velocity" in out["T4"]["rule_flags"]
