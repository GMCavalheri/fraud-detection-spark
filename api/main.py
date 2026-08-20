import json
import os
from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import db
import inference
from constants import CATEGORIES, CHANNELS, CITY_COORDS
from logging_config import RequestLoggingMiddleware, configure_logging
from schemas import (
    CategoryBreakdown, DailyStat, ScoreRequest, ScoreResponse, SummaryStats,
    Transaction, TransactionsPage,
)

PROCESSED_DATA_DIR = os.environ.get("PROCESSED_DATA_DIR", "/opt/data/processed")
METRICS_PATH = os.path.join(PROCESSED_DATA_DIR, "metrics.json")
DQ_REPORT_PATH = os.path.join(PROCESSED_DATA_DIR, "data_quality_report.json")

logger = configure_logging()

app = FastAPI(
    title="Fraud Detection API",
    description="Serves Spark-cleaned, MLlib-scored transactions and live fraud scoring.",
    version="1.0.0",
)

app.add_middleware(RequestLoggingMiddleware, logger=logger)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_metrics() -> Optional[dict]:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/meta")
def meta():
    return {"categories": CATEGORIES, "channels": CHANNELS, "cities": sorted(CITY_COORDS.keys())}


@app.get("/transactions", response_model=TransactionsPage)
def list_transactions(
    limit: int = Query(50, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    predicted_label: Optional[int] = Query(None, ge=0, le=1),
    category: Optional[str] = None,
    min_amount: Optional[float] = Query(None, ge=0),
    max_amount: Optional[float] = Query(None, ge=0),
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    account_id: Optional[str] = None,
):
    rows, total = db.fetch_transactions(
        limit=limit, offset=offset, predicted_label=predicted_label, category=category,
        min_amount=min_amount, max_amount=max_amount, start_date=start_date, end_date=end_date,
        account_id=account_id,
    )
    return {"total": total, "limit": limit, "offset": offset, "items": rows}


@app.get("/transactions/{transaction_id}", response_model=Transaction)
def get_transaction(transaction_id: str):
    row = db.fetch_transaction_by_id(transaction_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return row


@app.get("/stats/summary", response_model=SummaryStats)
def stats_summary():
    stats = db.fetch_summary_stats()
    if not stats or stats.get("total_transactions", 0) == 0:
        raise HTTPException(status_code=404, detail="No scored data yet - has the Spark pipeline run?")
    stats["model_metrics"] = _load_metrics()
    return stats


@app.get("/stats/timeseries", response_model=list[DailyStat])
def stats_timeseries():
    return db.fetch_timeseries()


@app.get("/stats/categories", response_model=list[CategoryBreakdown])
def stats_categories():
    return db.fetch_category_breakdown()


@app.get("/reports/data-quality")
def data_quality_report():
    if not os.path.exists(DQ_REPORT_PATH):
        raise HTTPException(status_code=404, detail="Data quality report not found - has etl_clean.py run?")
    with open(DQ_REPORT_PATH) as f:
        return json.load(f)


@app.post("/score", response_model=ScoreResponse)
def score_transaction(req: ScoreRequest):
    if req.category not in CATEGORIES:
        raise HTTPException(status_code=422, detail=f"category must be one of {CATEGORIES}")
    if req.channel not in CHANNELS:
        raise HTTPException(status_code=422, detail=f"channel must be one of {CHANNELS}")
    try:
        return inference.score(req)
    except Exception as exc:  # model/spark not ready, etc.
        logger.exception("Scoring failed for account_id=%s category=%s", req.account_id, req.category)
        raise HTTPException(status_code=503, detail=f"Scoring unavailable: {exc}")
