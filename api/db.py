import os
from datetime import date, datetime, timedelta
from typing import Optional

from sqlalchemy import create_engine, text

DATABASE_URL = (
    f"postgresql+psycopg2://{os.environ.get('POSTGRES_USER', 'fraud_admin')}:"
    f"{os.environ.get('POSTGRES_PASSWORD', 'change_me')}@"
    f"{os.environ.get('POSTGRES_HOST', 'postgres')}:"
    f"{os.environ.get('POSTGRES_PORT', '5432')}/"
    f"{os.environ.get('POSTGRES_DB', 'fraud_detection')}"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)


def fetch_transactions(
    limit: int = 50,
    offset: int = 0,
    predicted_label: Optional[int] = None,
    category: Optional[str] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    account_id: Optional[str] = None,
):
    clauses = []
    params = {"limit": limit, "offset": offset}
    if predicted_label is not None:
        clauses.append("predicted_label = :predicted_label")
        params["predicted_label"] = predicted_label
    if category:
        clauses.append("category = :category")
        params["category"] = category
    if min_amount is not None:
        clauses.append("amount_usd >= :min_amount")
        params["min_amount"] = min_amount
    if max_amount is not None:
        clauses.append("amount_usd <= :max_amount")
        params["max_amount"] = max_amount
    if start_date is not None:
        clauses.append("event_date >= :start_date")
        params["start_date"] = start_date
    if end_date is not None:
        clauses.append("event_date <= :end_date")
        params["end_date"] = end_date
    if account_id:
        clauses.append("account_id = :account_id")
        params["account_id"] = account_id

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = text(
        f"""
        SELECT transaction_id, source_system, account_id, card_last4, event_ts,
               event_date, amount_usd, currency_original, merchant, category,
               city, country, channel, is_fraud_actual, predicted_label,
               fraud_probability, rule_flags, txn_count_1h, txn_count_24h, amount_zscore
        FROM transactions_scored
        {where_sql}
        ORDER BY event_ts DESC
        LIMIT :limit OFFSET :offset
        """
    )
    count_query = text(f"SELECT count(*) FROM transactions_scored {where_sql}")

    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(query, params)]
        total = conn.execute(count_query, params).scalar_one()
    return rows, total


def fetch_transaction_by_id(transaction_id: str):
    query = text(
        """
        SELECT transaction_id, source_system, account_id, card_last4, event_ts,
               event_date, amount_usd, currency_original, merchant, category,
               city, country, channel, is_fraud_actual, predicted_label,
               fraud_probability, rule_flags, txn_count_1h, txn_count_24h, amount_zscore
        FROM transactions_scored
        WHERE transaction_id = :transaction_id
        LIMIT 1
        """
    )
    with engine.connect() as conn:
        row = conn.execute(query, {"transaction_id": transaction_id}).first()
    return dict(row._mapping) if row else None


def fetch_summary_stats():
    query = text(
        """
        SELECT
            count(*) AS total_transactions,
            sum(is_fraud_actual) AS actual_fraud_count,
            sum(predicted_label) AS predicted_fraud_count,
            sum(amount_usd) AS total_amount_usd,
            sum(CASE WHEN predicted_label = 1 THEN amount_usd ELSE 0 END) AS flagged_amount_usd,
            avg(fraud_probability) AS avg_fraud_probability,
            min(event_date) AS earliest_date,
            max(event_date) AS latest_date
        FROM transactions_scored
        """
    )
    with engine.connect() as conn:
        row = conn.execute(query).first()
    return dict(row._mapping) if row else {}


def fetch_timeseries():
    query = text(
        """
        SELECT event_date, total_transactions, actual_fraud_count,
               predicted_fraud_count, total_amount_usd, flagged_amount_usd,
               avg_fraud_probability
        FROM daily_stats
        ORDER BY event_date
        """
    )
    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(query)]
    return rows


def fetch_category_breakdown():
    query = text(
        """
        SELECT category,
               count(*) AS total_transactions,
               sum(predicted_label) AS predicted_fraud_count,
               sum(amount_usd) AS total_amount_usd
        FROM transactions_scored
        GROUP BY category
        ORDER BY predicted_fraud_count DESC
        """
    )
    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(query)]
    return rows


def fetch_account_recent_activity(account_id: str, as_of: datetime):
    """Pull the trailing 1h/24h velocity context and last known location for
    an existing account, used to build realistic features for the live
    /score endpoint instead of scoring in a vacuum."""
    window_query = text(
        """
        SELECT
            count(*) FILTER (WHERE event_ts >= :one_hour_ago) AS txn_count_1h,
            coalesce(sum(amount_usd) FILTER (WHERE event_ts >= :one_hour_ago), 0) AS txn_amount_sum_1h,
            count(*) FILTER (WHERE event_ts >= :one_day_ago) AS txn_count_24h,
            coalesce(sum(amount_usd) FILTER (WHERE event_ts >= :one_day_ago), 0) AS txn_amount_sum_24h,
            avg(amount_usd) AS hist_avg_amount,
            stddev(amount_usd) AS hist_stddev_amount,
            count(*) AS hist_txn_count
        FROM transactions_scored
        WHERE account_id = :account_id AND event_ts < :as_of
        """
    )
    last_txn_query = text(
        """
        SELECT event_ts, city, category
        FROM transactions_scored
        WHERE account_id = :account_id AND event_ts < :as_of
        ORDER BY event_ts DESC
        LIMIT 1
        """
    )
    params = {
        "account_id": account_id,
        "as_of": as_of,
        "one_hour_ago": as_of - timedelta(hours=1),
        "one_day_ago": as_of - timedelta(hours=24),
    }
    with engine.connect() as conn:
        stats_row = conn.execute(window_query, params).first()
        last_row = conn.execute(last_txn_query, params).first()
    stats = dict(stats_row._mapping) if stats_row else {}
    last_txn = dict(last_row._mapping) if last_row else None
    return stats, last_txn


def account_has_category(account_id: str, category: str, as_of: datetime) -> bool:
    query = text(
        """
        SELECT 1 FROM transactions_scored
        WHERE account_id = :account_id AND category = :category AND event_ts < :as_of
        LIMIT 1
        """
    )
    with engine.connect() as conn:
        row = conn.execute(query, {"account_id": account_id, "category": category, "as_of": as_of}).first()
    return row is not None
