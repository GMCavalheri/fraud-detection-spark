#!/usr/bin/env bash
# Orchestrates the full fraud-detection ETL pipeline as a sequence of
# spark-submit jobs against the Spark cluster: clean -> engineer features ->
# train model -> score + load into Postgres.
set -euo pipefail

SPARK_MASTER_URL="${SPARK_MASTER_URL:-spark://spark-master:7077}"
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_job() {
  local script="$1"
  echo "=================================================================="
  echo "Running ${script}"
  echo "=================================================================="
  spark-submit \
    --master "${SPARK_MASTER_URL}" \
    --packages org.postgresql:postgresql:42.7.3 \
    --conf spark.driver.memory="${SPARK_DRIVER_MEMORY:-2g}" \
    --conf spark.executor.memory="${SPARK_EXECUTOR_MEMORY:-2g}" \
    "${JOB_DIR}/${script}"
}

run_job "etl_clean.py"
run_job "feature_engineering.py"
run_job "train_model.py"
run_job "score_and_load.py"

echo "Pipeline complete."
