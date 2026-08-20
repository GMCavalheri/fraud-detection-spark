from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    transaction_id: str
    source_system: str
    account_id: str
    card_last4: Optional[str] = None
    event_ts: datetime
    event_date: date
    amount_usd: float
    currency_original: Optional[str] = None
    merchant: Optional[str] = None
    category: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    channel: Optional[str] = None
    is_fraud_actual: Optional[int] = None
    predicted_label: Optional[int] = None
    fraud_probability: Optional[float] = None
    rule_flags: Optional[str] = None
    txn_count_1h: Optional[int] = None
    txn_count_24h: Optional[int] = None
    amount_zscore: Optional[float] = None


class TransactionsPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: List[Transaction]


class SummaryStats(BaseModel):
    total_transactions: int
    actual_fraud_count: int
    predicted_fraud_count: int
    total_amount_usd: float
    flagged_amount_usd: float
    avg_fraud_probability: Optional[float]
    earliest_date: Optional[date]
    latest_date: Optional[date]
    model_metrics: Optional[dict] = None


class DailyStat(BaseModel):
    event_date: date
    total_transactions: int
    actual_fraud_count: int
    predicted_fraud_count: int
    total_amount_usd: float
    flagged_amount_usd: float
    avg_fraud_probability: Optional[float]


class CategoryBreakdown(BaseModel):
    category: str
    total_transactions: int
    predicted_fraud_count: int
    total_amount_usd: float


class ScoreRequest(BaseModel):
    account_id: Optional[str] = Field(
        default=None, description="Existing account id to pull real recent history for (e.g. ACCT-116213). Leave blank to simulate a brand-new card."
    )
    amount_usd: float = Field(..., gt=0)
    category: str
    channel: str = "online"
    city: Optional[str] = None
    event_ts: Optional[datetime] = None


class ScoreResponse(BaseModel):
    fraud_probability: float
    predicted_label: int
    rule_flags: List[str]
    used_account_history: bool
    features_used: dict
