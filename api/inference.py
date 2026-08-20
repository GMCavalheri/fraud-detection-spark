"""
Live fraud scoring for the /score endpoint. Loads the exact PipelineModel
trained by spark_jobs/train_model.py (via a lightweight local SparkSession)
so the API scores with the same model the cluster produced, rather than a
re-implemented copy.

If an account_id is supplied, recent history is pulled from Postgres to
build realistic velocity/z-score/geo features; otherwise the transaction is
scored as a brand-new card with no history (all history features neutral).
"""

import logging
import math
import os
import time
from datetime import datetime, timezone

from constants import CITY_COORDS

import db

logger = logging.getLogger("api")

S3_BUCKET = os.environ.get("MINIO_BUCKET", "fraud-detection")
MODEL_PATH = os.environ.get("MODEL_PATH", f"s3a://{S3_BUCKET}/models/fraud_model")

_spark = None
_model = None


def _get_model():
    global _spark, _model
    if _model is None:
        logger.info("Loading fraud model from %s (first /score call - this takes a few seconds)", MODEL_PATH)
        start = time.monotonic()
        from pyspark.ml import PipelineModel
        from pyspark.sql import SparkSession

        _spark = (
            SparkSession.builder.appName("fraud-api-inference")
            .master("local[2]")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.ui.enabled", "false")
            .config(
                "spark.jars.packages",
                "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262",
            )
            .config("spark.hadoop.fs.s3a.endpoint", os.environ.get("MINIO_ENDPOINT", "http://minio:9000"))
            .config("spark.hadoop.fs.s3a.access.key", os.environ.get("MINIO_ACCESS_KEY", "minioadmin"))
            .config("spark.hadoop.fs.s3a.secret.key", os.environ.get("MINIO_SECRET_KEY", "minioadmin"))
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
            .getOrCreate()
        )
        _model = PipelineModel.load(MODEL_PATH)
        logger.info("Model loaded in %.1fs", time.monotonic() - start)
    return _spark, _model


def _haversine_km(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    r = 6371.0
    lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return r * 2 * math.asin(math.sqrt(a))


def _spark_day_of_week(dt: datetime) -> int:
    # Spark's dayofweek(): 1=Sunday ... 7=Saturday. Python isoweekday(): 1=Monday ... 7=Sunday.
    return (dt.isoweekday() % 7) + 1


def build_features(req) -> tuple[dict, bool]:
    event_ts = req.event_ts or datetime.now(timezone.utc)

    used_history = False
    txn_count_1h = txn_count_24h = 0
    txn_amount_sum_1h = txn_amount_sum_24h = 0.0
    amount_zscore = 0.0
    seconds_since_last_txn = -1
    distance_from_last_txn_km = -1.0
    is_new_merchant = 1

    if req.account_id:
        stats, last_txn = db.fetch_account_recent_activity(req.account_id, event_ts)
        if stats and stats.get("hist_txn_count", 0):
            used_history = True
            txn_count_1h = int(stats["txn_count_1h"])
            txn_count_24h = int(stats["txn_count_24h"])
            txn_amount_sum_1h = float(stats["txn_amount_sum_1h"])
            txn_amount_sum_24h = float(stats["txn_amount_sum_24h"])
            hist_avg = stats.get("hist_avg_amount")
            hist_std = stats.get("hist_stddev_amount")
            if hist_std:
                amount_zscore = (req.amount_usd - float(hist_avg)) / float(hist_std)

            is_new_merchant = 0 if db.account_has_category(req.account_id, req.category, event_ts) else 1

            if last_txn:
                seconds_since_last_txn = (event_ts.replace(tzinfo=None) - last_txn["event_ts"]).total_seconds()
                if req.city and last_txn.get("city") and req.city in CITY_COORDS and last_txn["city"] in CITY_COORDS:
                    distance_from_last_txn_km = _haversine_km(CITY_COORDS[req.city], CITY_COORDS[last_txn["city"]])

    # this transaction itself falls inside its own trailing windows
    txn_count_1h += 1
    txn_count_24h += 1
    txn_amount_sum_1h += req.amount_usd
    txn_amount_sum_24h += req.amount_usd

    implied_travel_speed_kmh = -1.0
    if distance_from_last_txn_km >= 0 and seconds_since_last_txn > 0:
        implied_travel_speed_kmh = distance_from_last_txn_km / (seconds_since_last_txn / 3600.0)
    impossible_travel_flag = 1 if implied_travel_speed_kmh > 900 else 0

    hour_of_day = event_ts.hour
    is_odd_hour = 1 if 1 <= hour_of_day <= 4 else 0
    rule_flag_high_velocity = 1 if txn_count_1h >= 5 else 0
    rule_flag_amount_outlier = 1 if abs(amount_zscore) >= 3 else 0

    features = {
        "amount_usd": float(req.amount_usd),
        "txn_count_1h": txn_count_1h,
        "txn_amount_sum_1h": txn_amount_sum_1h,
        "txn_count_24h": txn_count_24h,
        "txn_amount_sum_24h": txn_amount_sum_24h,
        "amount_zscore": float(amount_zscore),
        "seconds_since_last_txn": float(seconds_since_last_txn),
        "distance_from_last_txn_km": float(distance_from_last_txn_km),
        "implied_travel_speed_kmh": float(implied_travel_speed_kmh),
        "impossible_travel_flag": impossible_travel_flag,
        "is_new_merchant": is_new_merchant,
        "hour_of_day": hour_of_day,
        "day_of_week": _spark_day_of_week(event_ts),
        "is_odd_hour": is_odd_hour,
        "rule_flag_high_velocity": rule_flag_high_velocity,
        "rule_flag_amount_outlier": rule_flag_amount_outlier,
        "category": req.category,
        "channel": req.channel,
    }

    rule_flags = []
    if rule_flag_high_velocity:
        rule_flags.append("high_velocity")
    if rule_flag_amount_outlier:
        rule_flags.append("amount_outlier")
    if impossible_travel_flag:
        rule_flags.append("impossible_travel")
    if is_odd_hour:
        rule_flags.append("odd_hour")
    if is_new_merchant:
        rule_flags.append("new_merchant")

    return features, used_history, rule_flags


def score(req) -> dict:
    features, used_history, rule_flags = build_features(req)
    spark, model = _get_model()

    row_df = spark.createDataFrame([features])
    prediction = model.transform(row_df).select("prediction", "probability").first()
    fraud_probability = float(prediction["probability"][1])
    predicted_label = int(prediction["prediction"])

    logger.info(
        "Scored account_id=%s amount=%.2f category=%s -> probability=%.4f label=%d flags=%s",
        req.account_id, req.amount_usd, req.category, fraud_probability, predicted_label, rule_flags,
    )

    return {
        "fraud_probability": round(fraud_probability, 5),
        "predicted_label": predicted_label,
        "rule_flags": rule_flags,
        "used_account_history": used_history,
        "features_used": features,
    }
