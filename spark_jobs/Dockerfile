FROM bitnami/spark:3.5

USER root

WORKDIR /opt/spark_jobs

COPY common.py etl_clean.py feature_engineering.py train_model.py score_and_load.py run_pipeline.sh ./
RUN chmod +x run_pipeline.sh

ENTRYPOINT ["./run_pipeline.sh"]
