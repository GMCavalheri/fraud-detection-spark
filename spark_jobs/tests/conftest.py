import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    # A plain local session, deliberately not common.get_spark(): the
    # functions under test don't touch Postgres or S3A, so skip pulling
    # those packages and keep the test suite fast/offline-friendly.
    session = (
        SparkSession.builder.appName("fraud-detection-tests")
        .master("local[2]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
    yield session
    session.stop()
