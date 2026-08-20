"""
Spark ETL step 1: ingest the three messy raw source systems, unify their
divergent schemas into one canonical transaction table, clean the data, and
write it out as partitioned Parquet plus a data-quality report.

Run with:
    spark-submit etl_clean.py
"""

import json
import os

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StringType, StructType, StructField, DoubleType,
)

from common import (
    CANONICAL_COLUMNS, CITY_COORDS, CLEANED_PATH, CURRENCY_RATES_TO_USD,
    DQ_REPORT_PATH, MCC_TO_CATEGORY, RAW_DIR, get_logger, get_spark,
)

logger = get_logger("etl_clean")

MOBILE_SCHEMA = StructType([
    StructField("evt_id", StringType()),
    StructField("acct_card", StringType()),
    StructField("event_ts", StringType()),
    StructField("amt", DoubleType()),
    StructField("amt_currency", StringType()),
    StructField("merchant", StringType()),
    StructField("merchant_category", StringType()),
    StructField("device", StructType([
        StructField("lat", DoubleType()),
        StructField("lon", DoubleType()),
        StructField("os", StringType()),
    ])),
    StructField("city_name", StringType()),
    StructField("is_fraud", StringType()),
])


def read_core(spark, raw_dir=None):
    raw_dir = raw_dir or RAW_DIR
    df = spark.read.option("header", True).csv(os.path.join(raw_dir, "core_transactions"))
    raw_count = df.count()

    ts_iso = F.to_timestamp("txn_date", "yyyy-MM-dd'T'HH:mm:ss")
    ts_us = F.to_timestamp("txn_date", "MM/dd/yyyy hh:mm a")
    ts_unix = F.when(
        F.col("txn_date").rlike("^[0-9]+$"), F.col("txn_date").cast("long").cast("timestamp")
    )
    event_ts = F.coalesce(ts_iso, ts_us, ts_unix)

    amount_clean = F.regexp_replace("amount", "[$,]", "").cast("double")

    out = df.select(
        F.col("transaction_id"),
        F.lit("core_transactions").alias("source_system"),
        F.col("account_id"),
        F.col("customer_name"),
        F.substring(F.col("card_number"), -4, 4).alias("card_last4"),
        F.substring(F.sha2(F.col("card_number"), 256), 1, 12).alias("card_hash"),
        event_ts.alias("event_ts"),
        F.abs(amount_clean).alias("amount_usd"),
        amount_clean.alias("amount_original"),
        F.lit("USD").alias("currency_original"),
        F.col("merchant_name").alias("merchant"),
        F.initcap(F.trim(F.col("category"))).alias("category"),
        F.col("city"),
        F.col("country"),
        F.lit(None).cast("double").alias("lat"),
        F.lit(None).cast("double").alias("lon"),
        F.col("channel"),
        F.col("is_fraud").cast("int").alias("is_fraud_label"),
        (amount_clean < 0).alias("_had_negative_amount"),
    )
    return out, raw_count


def read_mobile(spark, raw_dir=None):
    raw_dir = raw_dir or RAW_DIR
    raw_lines = spark.read.text(os.path.join(raw_dir, "mobile_events"))
    total_lines = raw_lines.count()

    df = spark.read.schema(MOBILE_SCHEMA).option("mode", "DROPMALFORMED").json(
        os.path.join(raw_dir, "mobile_events")
    )
    parsed_count = df.count()
    corrupt_lines = total_lines - parsed_count

    out = df.select(
        F.col("evt_id").alias("transaction_id"),
        F.lit("mobile_events").alias("source_system"),
        F.col("acct_card").alias("account_id"),
        F.lit(None).cast("string").alias("customer_name"),
        F.lit(None).cast("string").alias("card_last4"),
        F.lit(None).cast("string").alias("card_hash"),
        F.to_timestamp("event_ts", "yyyy-MM-dd'T'HH:mm:ssXX").alias("event_ts"),
        F.when(F.upper("amt_currency") == "USD", F.col("amt"))
         .otherwise(F.col("amt") * F.lit(CURRENCY_RATES_TO_USD.get("EUR", 1.0)))
         .alias("amount_usd"),
        F.col("amt").alias("amount_original"),
        F.upper(F.col("amt_currency")).alias("currency_original"),
        F.col("merchant"),
        F.initcap(F.trim(F.col("merchant_category"))).alias("category"),
        F.col("city_name").alias("city"),
        F.lit(None).cast("string").alias("country"),
        F.col("device.lat").alias("lat"),
        F.col("device.lon").alias("lon"),
        F.lit("online").alias("channel"),
        F.col("is_fraud").cast("int").alias("is_fraud_label"),
        F.lit(False).alias("_had_negative_amount"),
    )
    return out, total_lines, corrupt_lines


def read_legacy(spark, raw_dir=None):
    raw_dir = raw_dir or RAW_DIR
    df = spark.read.option("header", True).csv(os.path.join(raw_dir, "legacy_feed"))
    raw_count = df.count()

    event_ts = F.to_timestamp("DATE", "dd-MMM-yyyy HH:mm")
    rate_map = F.create_map([F.lit(x) for pair in CURRENCY_RATES_TO_USD.items() for x in pair])
    mcc_map = F.create_map([F.lit(x) for pair in MCC_TO_CATEGORY.items() for x in pair])
    amount_local = F.col("AMOUNT").cast("double")

    out = df.select(
        F.col("TXN_ID").alias("transaction_id"),
        F.lit("legacy_feed").alias("source_system"),
        F.col("ACCOUNT_REF").alias("account_id"),
        F.col("CUST_NAME").alias("customer_name"),
        F.substring(F.col("CARD_DISPLAY"), -4, 4).alias("card_last4"),
        F.substring(F.sha2(F.concat_ws("|", F.col("ACCOUNT_REF"), F.col("CARD_DISPLAY")), 256), 1, 12).alias("card_hash"),
        event_ts.alias("event_ts"),
        (amount_local * rate_map.getItem(F.col("CCY"))).alias("amount_usd"),
        amount_local.alias("amount_original"),
        F.col("CCY").alias("currency_original"),
        F.col("MERCHANT").alias("merchant"),
        F.initcap(mcc_map.getItem(F.col("MCC"))).alias("category"),
        F.lit(None).cast("string").alias("city"),
        F.col("COUNTRY").alias("country"),
        F.lit(None).cast("double").alias("lat"),
        F.lit(None).cast("double").alias("lon"),
        F.lit("pos").alias("channel"),
        F.col("IS_FRAUD").cast("int").alias("is_fraud_label"),
        F.lit(False).alias("_had_negative_amount"),
    )
    return out, raw_count


def geocode_missing_coords(df):
    """Fill lat/lon from a city-name lookup for rows that only carry a city
    (e.g. core_transactions has no native coordinates), leaving existing
    lat/lon untouched. Pure enough to unit-test without files."""
    lat_map = F.create_map([F.lit(x) for city, (lat, _lon) in CITY_COORDS.items() for x in (city, lat)])
    lon_map = F.create_map([F.lit(x) for city, (_lat, lon) in CITY_COORDS.items() for x in (city, lon)])
    return df.withColumn(
        "lat", F.coalesce(F.col("lat"), lat_map.getItem(F.col("city")))
    ).withColumn(
        "lon", F.coalesce(F.col("lon"), lon_map.getItem(F.col("city")))
    )


def main():
    logger.info("Starting etl_clean")
    spark = get_spark("fraud-etl-clean")
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

    core_df, core_raw = read_core(spark)
    logger.info("Read core_transactions: %d raw rows", core_raw)
    mobile_df, mobile_raw, mobile_corrupt = read_mobile(spark)
    logger.info("Read mobile_events: %d lines, %d corrupt lines dropped", mobile_raw, mobile_corrupt)
    legacy_df, legacy_raw = read_legacy(spark)
    logger.info("Read legacy_feed: %d raw rows", legacy_raw)

    negative_amount_count = core_df.filter(F.col("_had_negative_amount")).count()

    union_df = (
        core_df.drop("_had_negative_amount")
        .unionByName(mobile_df.drop("_had_negative_amount"))
        .unionByName(legacy_df.drop("_had_negative_amount"))
    )
    union_df = union_df.select(*CANONICAL_COLUMNS)

    # geocode rows that only carry a city name (e.g. core_transactions) so the
    # geo-velocity fraud feature has coordinates to work with
    union_df = geocode_missing_coords(union_df)

    total_before_dedup = union_df.count()

    deduped = union_df.dropDuplicates(["source_system", "transaction_id"])
    after_dedup = deduped.count()
    duplicates_removed = total_before_dedup - after_dedup
    logger.info("Deduplicated: %d rows -> %d rows (%d duplicates removed)", total_before_dedup, after_dedup, duplicates_removed)

    missing_merchant = deduped.filter(F.col("merchant").isNull()).count()
    missing_city = deduped.filter(F.col("city").isNull()).count()

    cleaned = deduped.withColumn(
        "merchant", F.coalesce(F.col("merchant"), F.lit("UNKNOWN"))
    ).withColumn(
        "city", F.coalesce(F.col("city"), F.lit("UNKNOWN"))
    ).withColumn(
        "category", F.coalesce(F.col("category"), F.lit("Other"))
    )

    now = spark.sql("select current_timestamp() as ts").collect()[0]["ts"]
    future_cutoff = F.lit(now) + F.expr("INTERVAL 1 DAY")
    invalid_ts = cleaned.filter(
        F.col("event_ts").isNull() | (F.col("event_ts") > future_cutoff)
    ).count()
    cleaned = cleaned.filter(
        F.col("event_ts").isNotNull() & (F.col("event_ts") <= future_cutoff)
    )

    final_count = cleaned.count()
    logger.info("Final cleaned row count: %d (%d invalid/future timestamps dropped)", final_count, invalid_ts)

    cleaned = cleaned.withColumn("event_date", F.to_date("event_ts"))
    logger.info("Writing cleaned Parquet to %s", CLEANED_PATH)
    cleaned.coalesce(4).write.mode("overwrite").partitionBy("event_date").parquet(CLEANED_PATH)

    report = {
        "raw_row_counts": {
            "core_transactions": core_raw,
            "mobile_events_lines": mobile_raw,
            "mobile_events_corrupt_lines_dropped": mobile_corrupt,
            "legacy_feed": legacy_raw,
        },
        "total_raw_rows": core_raw + mobile_raw + legacy_raw,
        "rows_after_union": total_before_dedup,
        "duplicates_removed": duplicates_removed,
        "negative_amounts_corrected": negative_amount_count,
        "missing_merchant_imputed": missing_merchant,
        "missing_city_imputed": missing_city,
        "invalid_or_future_timestamps_dropped": invalid_ts,
        "final_cleaned_rows": final_count,
    }

    os.makedirs(os.path.dirname(DQ_REPORT_PATH), exist_ok=True)
    with open(DQ_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Data quality report written to %s", DQ_REPORT_PATH)
    logger.info("Data quality report: %s", json.dumps(report))

    spark.stop()
    logger.info("etl_clean complete")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("etl_clean failed")
        raise
