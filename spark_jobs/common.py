"""Shared configuration and helpers for all Spark jobs in the fraud detection pipeline."""

import os

from pyspark.sql import SparkSession

RAW_DIR = os.environ.get("RAW_DATA_DIR", "/opt/data/raw")
PROCESSED_DIR = os.environ.get("PROCESSED_DATA_DIR", "/opt/data/processed")
MODELS_DIR = os.environ.get("MODELS_DIR", "/opt/models")

CLEANED_PATH = os.path.join(PROCESSED_DIR, "cleaned")
FEATURES_PATH = os.path.join(PROCESSED_DIR, "features")
DQ_REPORT_PATH = os.path.join(PROCESSED_DIR, "data_quality_report.json")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
MODEL_PATH = os.path.join(MODELS_DIR, "fraud_model")

# category taxonomy shared with the data generator so MCC codes / free-text
# categories from every source map onto the same canonical set
CATEGORIES = [
    "Electronics", "Groceries", "Restaurants", "Travel", "Fuel", "Fashion",
    "Entertainment", "Health & Pharmacy", "Home & Garden", "Online Services",
    "Utilities", "Jewelry",
]

MCC_TO_CATEGORY = {
    "5732": "Electronics", "5411": "Groceries", "5812": "Restaurants",
    "4511": "Travel", "5541": "Fuel", "5651": "Fashion", "7832": "Entertainment",
    "5912": "Health & Pharmacy", "5200": "Home & Garden", "5045": "Online Services",
    "4900": "Utilities", "5944": "Jewelry",
}

CURRENCY_RATES_TO_USD = {"USD": 1.0, "EUR": 1.08, "BRL": 0.18, "GBP": 1.27, "JPY": 0.0067}

# reference lookup so sources that only report a city name (no native lat/lon,
# e.g. core_transactions) can still be geocoded for the geo-velocity feature
CITY_COORDS = {
    "New York": (40.7128, -74.0060), "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298), "London": (51.5074, -0.1278),
    "Manchester": (53.4808, -2.2426), "Paris": (48.8566, 2.3522),
    "Berlin": (52.5200, 13.4050), "Madrid": (40.4168, -3.7038),
    "Sao Paulo": (-23.5505, -46.6333), "Rio de Janeiro": (-22.9068, -43.1729),
    "Brasilia": (-15.7939, -47.8828), "Tokyo": (35.6762, 139.6503),
    "Osaka": (34.6937, 135.5023), "Sydney": (-33.8688, 151.2093),
    "Melbourne": (-37.8136, 144.9631), "Toronto": (43.6532, -79.3832),
    "Vancouver": (49.2827, -123.1207), "Dubai": (25.2048, 55.2708),
    "Singapore": (1.3521, 103.8198), "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025), "Mexico City": (19.4326, -99.1332),
    "Lagos": (6.5244, 3.3792), "Johannesburg": (-26.2041, 28.0473),
    "Rome": (41.9028, 12.4964), "Amsterdam": (52.3676, 4.9041),
    "Lisbon": (38.7223, -9.1393), "Seoul": (37.5665, 126.9780),
    "Buenos Aires": (-34.6037, -58.3816), "Warsaw": (52.2297, 21.0122),
}

CANONICAL_COLUMNS = [
    "transaction_id", "source_system", "account_id", "customer_name",
    "card_last4", "card_hash", "event_ts", "amount_usd", "amount_original",
    "currency_original", "merchant", "category", "city", "country",
    "lat", "lon", "channel", "is_fraud_label",
]


def get_spark(app_name: str) -> SparkSession:
    master = os.environ.get("SPARK_MASTER_URL", "local[*]")
    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "16"))
        .config("spark.jars.packages", "org.postgresql:postgresql:42.7.3")
    )
    return builder.getOrCreate()


def postgres_config():
    host = os.environ.get("POSTGRES_HOST", "postgres")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "fraud_detection")
    user = os.environ.get("POSTGRES_USER", "fraud_admin")
    password = os.environ.get("POSTGRES_PASSWORD", "change_me")
    url = f"jdbc:postgresql://{host}:{port}/{db}"
    props = {"user": user, "password": password, "driver": "org.postgresql.Driver"}
    return url, props
