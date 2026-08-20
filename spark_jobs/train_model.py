"""
Spark ETL step 3: train a GBT fraud classifier on the engineered feature
table, using a time-based train/test split (train on earlier days, test on
the most recent days) to avoid look-ahead leakage. Saves the fitted Spark
MLlib PipelineModel and an evaluation report.

Run with:
    spark-submit train_model.py
"""

import json
import os

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from common import FEATURES_PATH, METRICS_PATH, MODEL_PATH, get_logger, get_spark

logger = get_logger("train_model")

NUMERIC_FEATURES = [
    "amount_usd", "txn_count_1h", "txn_amount_sum_1h", "txn_count_24h",
    "txn_amount_sum_24h", "amount_zscore", "seconds_since_last_txn",
    "distance_from_last_txn_km", "implied_travel_speed_kmh",
    "impossible_travel_flag", "is_new_merchant", "hour_of_day", "day_of_week",
    "is_odd_hour", "rule_flag_high_velocity", "rule_flag_amount_outlier",
]
CATEGORICAL_FEATURES = ["category", "channel"]
LABEL_COL = "is_fraud_label"
TEST_FRACTION_DAYS = 0.2


def main():
    logger.info("Starting train_model")
    spark = get_spark("fraud-model-training")
    df = spark.read.parquet(FEATURES_PATH).filter(F.col(LABEL_COL).isNotNull())
    logger.info("Read feature Parquet from %s", FEATURES_PATH)

    date_bounds = df.select(F.min("event_date").alias("min_d"), F.max("event_date").alias("max_d")).collect()[0]
    min_d, max_d = date_bounds["min_d"], date_bounds["max_d"]
    total_days = (max_d - min_d).days or 1
    cutoff = min_d + (max_d - min_d) * (1 - TEST_FRACTION_DAYS)

    train_df = df.filter(F.col("event_date") < F.lit(cutoff))
    test_df = df.filter(F.col("event_date") >= F.lit(cutoff))

    train_pos = train_df.filter(F.col(LABEL_COL) == 1).count()
    train_neg = train_df.filter(F.col(LABEL_COL) == 0).count()
    weight_pos = (train_neg / train_pos) if train_pos else 1.0
    train_df = train_df.withColumn(
        "class_weight", F.when(F.col(LABEL_COL) == 1, F.lit(weight_pos)).otherwise(F.lit(1.0))
    )

    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in CATEGORICAL_FEATURES
    ]
    feature_cols = NUMERIC_FEATURES + [f"{c}_idx" for c in CATEGORICAL_FEATURES]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
    gbt = GBTClassifier(
        labelCol=LABEL_COL, featuresCol="features", weightCol="class_weight",
        maxIter=50, maxDepth=5, seed=42,
    )
    pipeline = Pipeline(stages=indexers + [assembler, gbt])

    logger.info("Training on %d rows (< %s), testing on %d rows (>= %s)", train_df.count(), cutoff, test_df.count(), cutoff)
    model = pipeline.fit(train_df)
    logger.info("Model fit complete")

    predictions = model.transform(test_df).withColumn(
        "fraud_probability", vector_to_array(F.col("probability"))[1]
    )
    predictions.cache()

    evaluator_roc = BinaryClassificationEvaluator(labelCol=LABEL_COL, metricName="areaUnderROC")
    evaluator_pr = BinaryClassificationEvaluator(labelCol=LABEL_COL, metricName="areaUnderPR")
    auc_roc = evaluator_roc.evaluate(predictions)
    auc_pr = evaluator_pr.evaluate(predictions)

    counts = {
        (int(r[LABEL_COL]), int(r["prediction"])): r["count"]
        for r in predictions.groupBy(LABEL_COL, "prediction").count().collect()
    }
    tp = counts.get((1, 1), 0)
    fp = counts.get((0, 1), 0)
    tn = counts.get((0, 0), 0)
    fn = counts.get((1, 0), 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    gbt_model = model.stages[-1]
    importances = gbt_model.featureImportances.toArray()
    feature_importances = sorted(
        zip(feature_cols, [round(float(x), 5) for x in importances]),
        key=lambda x: -x[1],
    )

    metrics = {
        "train_rows": train_df.count(),
        "test_rows": test_df.count(),
        "train_date_range": [str(min_d), str(cutoff)],
        "test_date_range": [str(cutoff), str(max_d)],
        "fraud_rate_train": round(train_pos / train_df.count(), 5) if train_df.count() else 0,
        "fraud_rate_test": round(predictions.filter(F.col(LABEL_COL) == 1).count() / predictions.count(), 5),
        "auc_roc": round(auc_roc, 5),
        "auc_pr": round(auc_pr, 5),
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "f1": round(f1, 5),
        "confusion_matrix": {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn},
        "feature_importances": feature_importances,
        "feature_columns": feature_cols,
        "model_type": "GBTClassifier",
    }

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Metrics written to %s", METRICS_PATH)
    logger.info(
        "Evaluation: auc_roc=%.5f auc_pr=%.5f precision=%.5f recall=%.5f f1=%.5f",
        auc_roc, auc_pr, precision, recall, f1,
    )

    model.write().overwrite().save(MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)

    spark.stop()
    logger.info("train_model complete")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("train_model failed")
        raise
