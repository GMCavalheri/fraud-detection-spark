"""
Spark ETL step 2: derive fraud-signal features from the cleaned transaction
table using window functions over each account's transaction history
(velocity, amount z-score, geo-velocity / impossible travel, new-merchant).

Run with:
    spark-submit feature_engineering.py
"""

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from common import CLEANED_PATH, FEATURES_PATH, get_spark

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1_r, lon1_r, lat2_r, lon2_r = (F.radians(c) for c in (lat1, lon1, lat2, lon2))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = F.sin(dlat / 2) ** 2 + F.cos(lat1_r) * F.cos(lat2_r) * F.sin(dlon / 2) ** 2
    c = 2 * F.asin(F.sqrt(a))
    return EARTH_RADIUS_KM * c


def main():
    spark = get_spark("fraud-feature-engineering")
    df = spark.read.parquet(CLEANED_PATH)

    df = df.withColumn("event_ts_unix", F.unix_timestamp("event_ts"))

    order_by_time = Window.partitionBy("account_id").orderBy("event_ts_unix")
    range_1h = order_by_time.rangeBetween(-3600, 0)
    range_24h = order_by_time.rangeBetween(-86400, 0)
    history_only = order_by_time.rowsBetween(Window.unboundedPreceding, -1)

    df = df.withColumn("txn_count_1h", F.count("*").over(range_1h))
    df = df.withColumn("txn_amount_sum_1h", F.sum("amount_usd").over(range_1h))
    df = df.withColumn("txn_count_24h", F.count("*").over(range_24h))
    df = df.withColumn("txn_amount_sum_24h", F.sum("amount_usd").over(range_24h))

    df = df.withColumn("hist_avg_amount", F.avg("amount_usd").over(history_only))
    df = df.withColumn("hist_stddev_amount", F.stddev("amount_usd").over(history_only))
    df = df.withColumn(
        "amount_zscore",
        F.when(
            (F.col("hist_stddev_amount").isNotNull()) & (F.col("hist_stddev_amount") > 0),
            (F.col("amount_usd") - F.col("hist_avg_amount")) / F.col("hist_stddev_amount"),
        ).otherwise(F.lit(0.0)),
    )

    prev_ts = F.lag("event_ts_unix").over(order_by_time)
    prev_lat = F.lag("lat").over(order_by_time)
    prev_lon = F.lag("lon").over(order_by_time)

    df = df.withColumn("seconds_since_last_txn", F.col("event_ts_unix") - prev_ts)
    df = df.withColumn(
        "distance_from_last_txn_km",
        F.when(
            prev_lat.isNotNull() & prev_lon.isNotNull() & F.col("lat").isNotNull() & F.col("lon").isNotNull(),
            haversine_km(prev_lat, prev_lon, F.col("lat"), F.col("lon")),
        ),
    )
    df = df.withColumn(
        "implied_travel_speed_kmh",
        F.when(
            F.col("distance_from_last_txn_km").isNotNull() & (F.col("seconds_since_last_txn") > 0),
            F.col("distance_from_last_txn_km") / (F.col("seconds_since_last_txn") / F.lit(3600.0)),
        ),
    )
    # faster than any commercial flight -> physically impossible for the same card holder
    df = df.withColumn(
        "impossible_travel_flag",
        F.when(F.col("implied_travel_speed_kmh") > 900, F.lit(1)).otherwise(F.lit(0)),
    )

    merchant_order = Window.partitionBy("account_id", "merchant").orderBy("event_ts_unix")
    df = df.withColumn(
        "is_new_merchant",
        F.when(F.row_number().over(merchant_order) == 1, F.lit(1)).otherwise(F.lit(0)),
    )

    df = df.withColumn("hour_of_day", F.hour("event_ts"))
    df = df.withColumn("day_of_week", F.dayofweek("event_ts"))
    df = df.withColumn(
        "is_odd_hour", F.when((F.col("hour_of_day") >= 1) & (F.col("hour_of_day") <= 4), F.lit(1)).otherwise(F.lit(0))
    )

    # rule-based flags, independent of the trained model - shown in the API/dashboard
    # alongside the model's probability as an interpretable baseline
    df = df.withColumn(
        "rule_flag_high_velocity", F.when(F.col("txn_count_1h") >= 5, F.lit(1)).otherwise(F.lit(0))
    )
    df = df.withColumn(
        "rule_flag_amount_outlier", F.when(F.abs(F.col("amount_zscore")) >= 3, F.lit(1)).otherwise(F.lit(0))
    )
    df = df.withColumn(
        "rule_flags",
        F.array_join(
            F.filter(
                F.array(
                    F.when(F.col("rule_flag_high_velocity") == 1, F.lit("high_velocity")),
                    F.when(F.col("rule_flag_amount_outlier") == 1, F.lit("amount_outlier")),
                    F.when(F.col("impossible_travel_flag") == 1, F.lit("impossible_travel")),
                    F.when(F.col("is_odd_hour") == 1, F.lit("odd_hour")),
                    F.when(F.col("is_new_merchant") == 1, F.lit("new_merchant")),
                ),
                lambda x: x.isNotNull(),
            ),
            ",",
        ),
    )

    df = df.drop("hist_avg_amount", "hist_stddev_amount")
    df = df.fillna({"seconds_since_last_txn": -1, "distance_from_last_txn_km": -1.0, "implied_travel_speed_kmh": -1.0})

    df.write.mode("overwrite").partitionBy("event_date").parquet(FEATURES_PATH)
    print(f"Feature engineering complete. Rows written: {df.count()}")

    spark.stop()


if __name__ == "__main__":
    main()
