FROM bitnamilegacy/spark:3.5.6

USER root

# numpy/pandas/pyarrow are needed by pyspark.ml (GBTClassifier, vector_to_array);
# setuptools<72 provides the distutils shim pyspark.ml still imports on Python 3.12
RUN pip install --no-cache-dir "setuptools<72" "numpy<2" "pandas==2.2.2" "pyarrow==16.1.0"

WORKDIR /opt/spark_jobs

COPY common.py etl_clean.py feature_engineering.py train_model.py score_and_load.py run_pipeline.sh ./
RUN chmod +x run_pipeline.sh

ENTRYPOINT ["./run_pipeline.sh"]
