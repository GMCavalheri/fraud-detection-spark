-- Schema for the fraud detection serving layer.
-- Spark (score_and_load.py) truncates and reloads these tables on every
-- pipeline run; the API only ever reads from them.

CREATE TABLE IF NOT EXISTS transactions_scored (
    transaction_id      VARCHAR(64) NOT NULL,
    source_system       VARCHAR(32) NOT NULL,
    account_id          VARCHAR(32) NOT NULL,
    card_last4          VARCHAR(4),
    event_ts            TIMESTAMP NOT NULL,
    event_date          DATE NOT NULL,
    amount_usd          DOUBLE PRECISION NOT NULL,
    currency_original   VARCHAR(8),
    merchant             VARCHAR(255),
    category             VARCHAR(64),
    city                 VARCHAR(64),
    country               VARCHAR(64),
    channel               VARCHAR(16),
    is_fraud_actual      INTEGER,
    predicted_label       INTEGER,
    fraud_probability     DOUBLE PRECISION,
    rule_flags             VARCHAR(255),
    txn_count_1h          INTEGER,
    txn_count_24h         INTEGER,
    amount_zscore         DOUBLE PRECISION,
    PRIMARY KEY (source_system, transaction_id)
);

CREATE INDEX IF NOT EXISTS idx_txn_event_date ON transactions_scored (event_date);
CREATE INDEX IF NOT EXISTS idx_txn_predicted_label ON transactions_scored (predicted_label);
CREATE INDEX IF NOT EXISTS idx_txn_account ON transactions_scored (account_id);

CREATE TABLE IF NOT EXISTS daily_stats (
    event_date             DATE PRIMARY KEY,
    total_transactions     INTEGER NOT NULL,
    actual_fraud_count     INTEGER NOT NULL,
    predicted_fraud_count  INTEGER NOT NULL,
    total_amount_usd       DOUBLE PRECISION NOT NULL,
    flagged_amount_usd     DOUBLE PRECISION NOT NULL,
    avg_fraud_probability  DOUBLE PRECISION
);
