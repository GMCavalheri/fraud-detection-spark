"""
Spark ETL step 4: score every transaction with the trained fraud model,
combine it with the rule-based flags from feature engineering, and load the
results into Postgres for the FastAPI/Streamlit layer to serve.

Run with:
    spark-submit --packages org.postgresql:postgresql:42.7.3 score_and_load.py
"""

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from common import FEATURES_PATH, MODEL_PATH, get_logger, get_spark, postgres_config

logger = get_logger("score_and_load")


def main():
    logger.info("Starting score_and_load")
    spark = get_spark("fraud-score-and-load")
    df = spark.read.parquet(FEATURES_PATH)
    logger.info("Read feature Parquet from %s", FEATURES_PATH)

    model = PipelineModel.load(MODEL_PATH)
    logger.info("Loaded model from %s", MODEL_PATH)
    scored = model.transform(df).withColumn(
        "fraud_probability", vector_to_array(F.col("probability"))[1]
    )

    result = scored.select(
        F.col("transaction_id"),
        F.col("source_system"),
        F.col("account_id"),
        F.col("card_last4"),
        F.col("event_ts"),
        F.col("event_date"),
        F.col("amount_usd"),
        F.col("currency_original"),
        F.col("merchant"),
        F.col("category"),
        F.col("city"),
        F.col("country"),
        F.col("channel"),
        F.col("is_fraud_label").alias("is_fraud_actual"),
        F.col("prediction").cast("int").alias("predicted_label"),
        F.round(F.col("fraud_probability"), 5).alias("fraud_probability"),
        F.col("rule_flags"),
        F.col("txn_count_1h"),
        F.col("txn_count_24h"),
        F.round(F.col("amount_zscore"), 3).alias("amount_zscore"),
    )

    url, props = postgres_config()

    (
        result.write.mode("overwrite")
        .option("truncate", "true")
        .jdbc(url, "transactions_scored", properties=props)
    )
    logger.info("Loaded %d scored transactions into transactions_scored", result.count())

    daily_stats = (
        result.groupBy("event_date")
        .agg(
            F.count("*").alias("total_transactions"),
            F.sum("is_fraud_actual").alias("actual_fraud_count"),
            F.sum("predicted_label").alias("predicted_fraud_count"),
            F.sum(F.col("amount_usd")).alias("total_amount_usd"),
            F.sum(F.when(F.col("predicted_label") == 1, F.col("amount_usd")).otherwise(0.0)).alias("flagged_amount_usd"),
            F.avg("fraud_probability").alias("avg_fraud_probability"),
        )
        .orderBy("event_date")
    )

    (
        daily_stats.write.mode("overwrite")
        .option("truncate", "true")
        .jdbc(url, "daily_stats", properties=props)
    )
    logger.info("Loaded %d rows into daily_stats", daily_stats.count())

    spark.stop()
    logger.info("score_and_load complete")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("score_and_load failed")
        raise
