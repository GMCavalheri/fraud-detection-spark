"""
Generates synthetic, deliberately messy "multi-system" credit-card transaction data
for a Spark ETL + fraud detection pipeline.

Three fake source systems are produced, each with a different schema, so the
downstream Spark job has a real ETL problem to solve (schema drift, mixed date
formats, mixed currencies, dirty strings, duplicates, missing values):

  data/raw/core_transactions/*.csv   - main processing feed
  data/raw/mobile_events/*.json      - mobile app events (JSON lines, nested fields)
  data/raw/legacy_feed/*.csv         - old batch export (different field names/formats)

All three sources share the same underlying pool of `account_id`s (under different
column names) so cross-source velocity/fraud features are meaningful once the ETL
unifies them.

Usage:
    python generate_data.py --rows 5000000
    python generate_data.py --rows 50000   # fast smoke-test run
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import string

import numpy as np
import pandas as pd
from faker import Faker

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

N_ACCOUNTS = 120_000
DATE_RANGE_DAYS = 120
END_DATE = pd.Timestamp("2026-08-20T00:00:00Z")
START_DATE = END_DATE - pd.Timedelta(days=DATE_RANGE_DAYS)

SOURCE_SPLIT = {"core_transactions": 0.60, "mobile_events": 0.25, "legacy_feed": 0.15}
ROWS_PER_FILE = 300_000

FRAUD_RATE = 0.015  # approximate target, realized via the injection fractions below
POS_LABEL_FLIP_RATE = 0.15   # fraction of true-fraud rows mislabeled as legit (missed fraud)
NEG_LABEL_FLIP_RATE = 0.006  # fraction of legit rows mislabeled as fraud (false alarms)

CATEGORIES = [
    "Electronics", "Groceries", "Restaurants", "Travel", "Fuel", "Fashion",
    "Entertainment", "Health & Pharmacy", "Home & Garden", "Online Services",
    "Utilities", "Jewelry",
]

# legacy feed only stores an MCC code -> needs mapping downstream in ETL
MCC_MAP = {
    "5732": "Electronics", "5411": "Groceries", "5812": "Restaurants",
    "4511": "Travel", "5541": "Fuel", "5651": "Fashion", "7832": "Entertainment",
    "5912": "Health & Pharmacy", "5200": "Home & Garden", "5045": "Online Services",
    "4900": "Utilities", "5944": "Jewelry",
}
CATEGORY_TO_MCC = {v: k for k, v in MCC_MAP.items()}

CITIES = [
    # name, country, lat, lon
    ("New York", "USA", 40.7128, -74.0060),
    ("Los Angeles", "USA", 34.0522, -118.2437),
    ("Chicago", "USA", 41.8781, -87.6298),
    ("London", "UK", 51.5074, -0.1278),
    ("Manchester", "UK", 53.4808, -2.2426),
    ("Paris", "France", 48.8566, 2.3522),
    ("Berlin", "Germany", 52.5200, 13.4050),
    ("Madrid", "Spain", 40.4168, -3.7038),
    ("Sao Paulo", "Brazil", -23.5505, -46.6333),
    ("Rio de Janeiro", "Brazil", -22.9068, -43.1729),
    ("Brasilia", "Brazil", -15.7939, -47.8828),
    ("Tokyo", "Japan", 35.6762, 139.6503),
    ("Osaka", "Japan", 34.6937, 135.5023),
    ("Sydney", "Australia", -33.8688, 151.2093),
    ("Melbourne", "Australia", -37.8136, 144.9631),
    ("Toronto", "Canada", 43.6532, -79.3832),
    ("Vancouver", "Canada", 49.2827, -123.1207),
    ("Dubai", "UAE", 25.2048, 55.2708),
    ("Singapore", "Singapore", 1.3521, 103.8198),
    ("Mumbai", "India", 19.0760, 72.8777),
    ("Delhi", "India", 28.7041, 77.1025),
    ("Mexico City", "Mexico", 19.4326, -99.1332),
    ("Lagos", "Nigeria", 6.5244, 3.3792),
    ("Johannesburg", "South Africa", -26.2041, 28.0473),
    ("Rome", "Italy", 41.9028, 12.4964),
    ("Amsterdam", "Netherlands", 52.3676, 4.9041),
    ("Lisbon", "Portugal", 38.7223, -9.1393),
    ("Seoul", "South Korea", 37.5665, 126.9780),
    ("Buenos Aires", "Argentina", -34.6037, -58.3816),
    ("Warsaw", "Poland", 52.2297, 21.0122),
]
N_CITIES = len(CITIES)
CITY_NAMES = np.array([c[0] for c in CITIES])
CITY_COUNTRIES = np.array([c[1] for c in CITIES])
CITY_LAT = np.array([c[2] for c in CITIES])
CITY_LON = np.array([c[3] for c in CITIES])

# pairs of city indices that are geographically far apart (used for "impossible travel" fraud)
FAR_PAIRS = [(0, 11), (3, 18), (8, 0), (11, 3), (15, 9), (18, 5), (1, 6), (23, 16)]

CURRENCY_RATES_TO_USD = {"USD": 1.0, "EUR": 1.08, "BRL": 0.18, "GBP": 1.27, "JPY": 0.0067}
CHANNELS = ["online", "pos", "atm"]

fake = Faker()
Faker.seed(SEED)
rng = np.random.default_rng(SEED)
random.seed(SEED)


def make_account_pool(n_accounts):
    return np.array([f"ACCT-{100000 + i}" for i in range(n_accounts)])


ACCOUNTS = make_account_pool(N_ACCOUNTS)
# power-law activity weights so some accounts transact far more than others
ACCOUNT_WEIGHTS = rng.pareto(a=2.0, size=N_ACCOUNTS) + 0.1
ACCOUNT_WEIGHTS = ACCOUNT_WEIGHTS / ACCOUNT_WEIGHTS.sum()

CUSTOMER_POOL = np.array([fake.name() for _ in range(5000)])
MERCHANT_POOL_BY_CATEGORY = {
    cat: np.array([fake.company() for _ in range(80)]) for cat in CATEGORIES
}


def random_accounts(n):
    return rng.choice(ACCOUNTS, size=n, p=ACCOUNT_WEIGHTS)


def random_timestamps(n):
    start_s = int(START_DATE.timestamp())
    end_s = int(END_DATE.timestamp())
    secs = rng.integers(start_s, end_s, size=n)
    return pd.to_datetime(secs, unit="s", utc=True)


def random_amounts(n, mean_log=3.4, sigma_log=1.0):
    amt = rng.lognormal(mean=mean_log, sigma=sigma_log, size=n)
    return np.round(amt, 2)


def random_categories(n):
    return rng.choice(CATEGORIES, size=n)


def merchants_for_categories(categories):
    out = np.empty(len(categories), dtype=object)
    for cat in CATEGORIES:
        mask = categories == cat
        cnt = mask.sum()
        if cnt:
            out[mask] = rng.choice(MERCHANT_POOL_BY_CATEGORY[cat], size=cnt)
    return out


def random_city_idx(n):
    return rng.integers(0, N_CITIES, size=n)


def hashed_last4(raw_id_series):
    return raw_id_series.apply(
        lambda s: hashlib.sha256(s.encode()).hexdigest()[:12]
    )


def fake_pan(n):
    return np.array(
        ["".join(random.choices(string.digits, k=16)) for _ in range(n)]
    )


def inject_missing(series, frac, fill=None):
    n = len(series)
    idx = rng.choice(n, size=int(n * frac), replace=False)
    series = series.copy()
    series[idx] = fill
    return series


def burst_rows(n_bursts, min_k, max_k, jitter_seconds):
    """Vectorized generation of clustered 'card testing' style bursts:
    a handful of accounts each firing several small transactions in a short window."""
    accounts = random_accounts(n_bursts)
    counts = rng.integers(min_k, max_k + 1, size=n_bursts)
    base_times = random_timestamps(n_bursts).astype("int64") // 10**9  # seconds

    rep_accounts = np.repeat(accounts, counts)
    rep_base = np.repeat(base_times, counts)
    jitter = rng.integers(0, jitter_seconds, size=rep_accounts.shape[0])
    ts = pd.to_datetime(rep_base + jitter, unit="s", utc=True)
    return rep_accounts, ts


def impossible_travel_rows(n_pairs, min_gap_min=10, max_gap_min=45):
    """Vectorized generation of account pairs of transactions in far-apart
    cities within an impossibly short time window."""
    accounts = random_accounts(n_pairs)
    pair_idx = rng.integers(0, len(FAR_PAIRS), size=n_pairs)
    city_a = np.array([FAR_PAIRS[i][0] for i in pair_idx])
    city_b = np.array([FAR_PAIRS[i][1] for i in pair_idx])
    t1 = random_timestamps(n_pairs)
    gap = rng.integers(min_gap_min * 60, max_gap_min * 60, size=n_pairs)
    t2 = t1 + pd.to_timedelta(gap, unit="s")

    accounts_all = np.concatenate([accounts, accounts])
    city_all = np.concatenate([city_a, city_b])
    ts_all = t1.append(t2) if hasattr(t1, "append") else pd.DatetimeIndex(np.concatenate([t1, t2]))
    return accounts_all, city_all, ts_all


def apply_label_noise(is_fraud):
    """Asymmetric label noise: some true fraud goes unlabeled (missed), and a
    much smaller share of legit transactions get mislabeled as fraud (false
    alarms) - keeps the overall rate near FRAUD_RATE while making the ML
    problem non-trivial."""
    is_fraud = is_fraud.copy()
    pos_idx = np.flatnonzero(is_fraud == 1)
    neg_idx = np.flatnonzero(is_fraud == 0)
    flip_pos = rng.choice(pos_idx, size=int(len(pos_idx) * POS_LABEL_FLIP_RATE), replace=False)
    flip_neg = rng.choice(neg_idx, size=int(len(neg_idx) * NEG_LABEL_FLIP_RATE), replace=False)
    is_fraud[flip_pos] = 0
    is_fraud[flip_neg] = 1
    return is_fraud


# --------------------------------------------------------------------------------------
# Source 1: core_transactions (CSV) - main feed
# --------------------------------------------------------------------------------------
MIN_CHUNK_FOR_PATTERNS = 2000  # below this, skip fraud-pattern injection entirely (avoids
                                # the "max(1, ...) floor exceeds a tiny trailing chunk" bug)


def generate_core_chunk(n, id_offset=0):
    if n < MIN_CHUNK_FOR_PATTERNS:
        n_burst = 0
        n_travel_pairs = 0
    else:
        n_burst = max(1, int(n * 0.0005))       # ~10 rows/burst -> ~0.5% of rows
        n_travel_pairs = max(1, int(n * 0.001))  # 2 rows/pair -> ~0.2% of rows
    n_background = n - n_burst * 10 - n_travel_pairs * 2  # rough budget, corrected after

    accounts_bg = random_accounts(n_background)
    ts_bg = random_timestamps(n_background)
    amt_bg = random_amounts(n_background)
    cat_bg = random_categories(n_background)
    city_idx_bg = random_city_idx(n_background)
    is_fraud_bg = np.zeros(n_background, dtype=int)

    burst_acc, burst_ts = burst_rows(n_burst, 5, 15, jitter_seconds=180)
    nb = len(burst_acc)
    burst_amt = np.round(rng.uniform(0.5, 4.5, size=nb), 2)
    burst_cat = rng.choice(["Online Services", "Electronics"], size=nb)
    burst_city_idx = random_city_idx(nb)
    burst_is_fraud = np.ones(nb, dtype=int)

    travel_acc, travel_city_idx, travel_ts = impossible_travel_rows(n_travel_pairs)
    nt = len(travel_acc)
    travel_amt = random_amounts(nt, mean_log=4.0)
    travel_cat = rng.choice(["Travel", "Electronics", "Jewelry"], size=nt)
    travel_is_fraud = np.ones(nt, dtype=int)

    # dormant-then-high-value + odd-hour-large: sampled from the background pool
    n_dormant = max(1, int(n_background * 0.0015))
    dormant_idx = rng.choice(n_background, size=n_dormant, replace=False)
    amt_bg[dormant_idx] = random_amounts(n_dormant, mean_log=8.2, sigma_log=0.4)
    is_fraud_bg[dormant_idx] = 1

    n_odd_hour = max(1, int(n_background * 0.0015))
    odd_idx = rng.choice(n_background, size=n_odd_hour, replace=False)
    odd_ts = ts_bg[odd_idx].floor("D") + pd.to_timedelta(
        rng.integers(2, 4, size=n_odd_hour), unit="h"
    ) + pd.to_timedelta(rng.integers(0, 59, size=n_odd_hour), unit="m")
    ts_bg = ts_bg.to_numpy()
    ts_bg[odd_idx] = odd_ts.to_numpy()
    amt_bg[odd_idx] = random_amounts(n_odd_hour, mean_log=7.5, sigma_log=0.5)
    is_fraud_bg[odd_idx] = 1

    accounts = np.concatenate([accounts_bg, burst_acc, travel_acc])
    ts = pd.DatetimeIndex(np.concatenate([ts_bg, burst_ts.to_numpy(), travel_ts.to_numpy()]))
    amt = np.concatenate([amt_bg, burst_amt, travel_amt])
    cat = np.concatenate([cat_bg, burst_cat, travel_cat])
    city_idx = np.concatenate([city_idx_bg, burst_city_idx, travel_city_idx])
    is_fraud = np.concatenate([is_fraud_bg, burst_is_fraud, travel_is_fraud])

    n_total = len(accounts)
    merchant = merchants_for_categories(cat)
    customer = rng.choice(CUSTOMER_POOL, size=n_total)
    pan = fake_pan(n_total)
    txn_id = np.array([f"CORE-{id_offset + i:09d}" for i in range(n_total)])
    channel = rng.choice(CHANNELS, size=n_total, p=[0.55, 0.4, 0.05])

    df = pd.DataFrame({
        "transaction_id": txn_id,
        "card_number": pan,
        "account_id": accounts,
        "customer_name": customer,
        "txn_date": ts,
        "amount": amt,
        "merchant_name": merchant,
        "category": cat,
        "city": CITY_NAMES[city_idx],
        "country": CITY_COUNTRIES[city_idx],
        "channel": channel,
        "is_fraud": is_fraud,
    })

    # --- dirty it up ---
    df["is_fraud"] = apply_label_noise(df["is_fraud"].to_numpy())

    # mixed date formats: split into 3 stylistic groups
    fmt_roll = rng.integers(0, 3, size=len(df))
    dt = df["txn_date"]
    iso_fmt = dt.dt.strftime("%Y-%m-%dT%H:%M:%S")
    us_fmt = dt.dt.strftime("%m/%d/%Y %I:%M %p")
    unix_fmt = (dt.astype("int64") // 10**9).astype(str)
    df["txn_date"] = np.select([fmt_roll == 0, fmt_roll == 1], [iso_fmt, us_fmt], default=unix_fmt)

    # amount as dirty string: $, commas, occasional negative (data entry errors)
    amt_str = df["amount"].map(lambda v: f"${v:,.2f}")
    neg_idx = rng.choice(len(df), size=int(len(df) * 0.002), replace=False)
    amt_arr = amt_str.to_numpy()
    amt_arr[neg_idx] = [f"-{v}" for v in amt_arr[neg_idx]]
    df["amount"] = amt_arr

    # inconsistent category casing
    case_roll = rng.integers(0, 3, size=len(df))
    cat_vals = df["category"].to_numpy(dtype=object)
    upper_idx = case_roll == 1
    lower_idx = case_roll == 2
    cat_vals[upper_idx] = [str(v).upper() for v in cat_vals[upper_idx]]
    cat_vals[lower_idx] = [str(v).lower() for v in cat_vals[lower_idx]]
    df["category"] = cat_vals

    # missing merchant / city
    df["merchant_name"] = inject_missing(df["merchant_name"], 0.03, fill=None)
    df["city"] = inject_missing(df["city"], 0.02, fill=None)

    # exact duplicate rows (simulate upstream retry/replay)
    dup_frac = 0.01
    dup_sample = df.sample(frac=dup_frac, random_state=int(rng.integers(0, 1_000_000)))
    df = pd.concat([df, dup_sample], ignore_index=True)

    return df.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Source 2: mobile_events (JSON lines) - schema drift, nested device info
# --------------------------------------------------------------------------------------
def generate_mobile_chunk(n, id_offset=0):
    if n < MIN_CHUNK_FOR_PATTERNS:
        n_burst = 0
    else:
        n_burst = max(1, int(n * 0.0008))  # ~13 rows/burst -> ~1% of rows
    n_background = n - n_burst * 13

    accounts_bg = random_accounts(n_background)
    ts_bg = random_timestamps(n_background)
    amt_bg = random_amounts(n_background, mean_log=3.2)
    cat_bg = random_categories(n_background)
    city_idx_bg = random_city_idx(n_background)
    is_fraud_bg = np.zeros(n_background, dtype=int)

    burst_acc, burst_ts = burst_rows(n_burst, 6, 20, jitter_seconds=120)
    nb = len(burst_acc)
    burst_amt = np.round(rng.uniform(0.5, 5.0, size=nb), 2)
    burst_cat = rng.choice(["Online Services", "Entertainment"], size=nb)
    burst_city_idx = random_city_idx(nb)
    burst_is_fraud = np.ones(nb, dtype=int)

    accounts = np.concatenate([accounts_bg, burst_acc])
    ts = pd.DatetimeIndex(np.concatenate([ts_bg.to_numpy(), burst_ts.to_numpy()]))
    amt = np.concatenate([amt_bg, burst_amt])
    cat = np.concatenate([cat_bg, burst_cat])
    city_idx = np.concatenate([city_idx_bg, burst_city_idx])
    is_fraud = np.concatenate([is_fraud_bg, burst_is_fraud])
    is_fraud = apply_label_noise(is_fraud)

    n_total = len(accounts)
    merchant = merchants_for_categories(cat)
    evt_id = np.array([f"MOB-{id_offset + i:09d}" for i in range(n_total)])
    os_choice = rng.choice(["iOS", "Android"], size=n_total)
    currency = rng.choice(["USD", "usd", "EUR"], size=n_total, p=[0.85, 0.1, 0.05])

    lat_jitter = rng.normal(0, 0.05, size=n_total)
    lon_jitter = rng.normal(0, 0.05, size=n_total)

    records = []
    ts_iso = ts.strftime("%Y-%m-%dT%H:%M:%S%z")
    for i in range(n_total):
        rec = {
            "evt_id": evt_id[i],
            "acct_card": accounts[i],
            "event_ts": ts_iso[i],
            "amt": round(float(amt[i]), 2),
            "amt_currency": currency[i],
            "merchant": merchant[i] if rng.random() > 0.01 else None,
            "merchant_category": cat[i],
            "device": {
                "lat": round(float(CITY_LAT[city_idx[i]] + lat_jitter[i]), 4),
                "lon": round(float(CITY_LON[city_idx[i]] + lon_jitter[i]), 4),
                "os": os_choice[i],
            },
            "city_name": CITY_NAMES[city_idx[i]],
            "is_fraud": int(is_fraud[i]),
        }
        records.append(rec)
    return records


# --------------------------------------------------------------------------------------
# Source 3: legacy_feed (CSV) - old batch export, different formats/currencies
# --------------------------------------------------------------------------------------
def generate_legacy_chunk(n, id_offset=0):
    accounts = random_accounts(n)
    ts = random_timestamps(n)
    amt_usd = random_amounts(n, mean_log=3.6)
    cat = random_categories(n)
    city_idx = random_city_idx(n)
    is_fraud = np.zeros(n, dtype=int)

    n_odd = max(1, int(n * 0.01))
    odd_idx = rng.choice(n, size=n_odd, replace=False)
    amt_usd[odd_idx] = random_amounts(n_odd, mean_log=7.8, sigma_log=0.5)
    is_fraud[odd_idx] = 1
    is_fraud = apply_label_noise(is_fraud)

    merchant = merchants_for_categories(cat)
    customer = rng.choice(CUSTOMER_POOL, size=n)
    txn_id = np.array([f"LEG-{id_offset + i:09d}" for i in range(n)])

    currency = rng.choice(["USD", "EUR", "BRL", "GBP"], size=n, p=[0.7, 0.15, 0.1, 0.05])
    rates = np.array([CURRENCY_RATES_TO_USD[c] for c in currency])
    amt_local = np.round(amt_usd / rates, 2)

    date_str = pd.DatetimeIndex(ts).strftime("%d-%b-%Y %H:%M")
    # out-of-range / future timestamps (data entry typo -> year 2099)
    future_idx = rng.choice(n, size=int(n * 0.001), replace=False)
    date_arr = date_str.to_numpy(dtype=object)
    date_arr[future_idx] = pd.DatetimeIndex(ts[future_idx]).strftime("%d-%b-2099 %H:%M")

    mcc = np.array([CATEGORY_TO_MCC[c] for c in cat])
    card_display = np.array([f"****{''.join(random.choices(string.digits, k=4))}" for _ in range(n)])

    df = pd.DataFrame({
        "TXN_ID": txn_id,
        "ACCOUNT_REF": accounts,
        "CARD_DISPLAY": card_display,
        "CUST_NAME": customer,
        "DATE": date_arr,
        "AMOUNT": amt_local,
        "CCY": currency,
        "MERCHANT": merchant,
        "MCC": mcc,
        "COUNTRY": CITY_COUNTRIES[city_idx],
        "IS_FRAUD": is_fraud,
    })

    df["MERCHANT"] = inject_missing(df["MERCHANT"], 0.025, fill=None)
    return df


def write_csv_partitions(gen_fn, total_rows, out_subdir, rows_per_file=ROWS_PER_FILE):
    out_path = os.path.join(OUT_DIR, out_subdir)
    os.makedirs(out_path, exist_ok=True)
    written = 0
    part = 0
    while written < total_rows:
        n = min(rows_per_file, total_rows - written)
        df = gen_fn(n, id_offset=written)
        df.to_csv(os.path.join(out_path, f"part-{part:04d}.csv"), index=False)
        written += n
        part += 1
        print(f"  [{out_subdir}] wrote part-{part:04d}.csv ({len(df)} rows, {written}/{total_rows})")


def write_json_partitions(gen_fn, total_rows, out_subdir, rows_per_file=ROWS_PER_FILE):
    out_path = os.path.join(OUT_DIR, out_subdir)
    os.makedirs(out_path, exist_ok=True)
    written = 0
    part = 0
    while written < total_rows:
        n = min(rows_per_file, total_rows - written)
        records = gen_fn(n, id_offset=written)
        file_path = os.path.join(out_path, f"part-{part:04d}.json")
        with open(file_path, "w") as f:
            for i, rec in enumerate(records):
                line = json.dumps(rec)
                # sprinkle a tiny number of corrupt lines to test permissive JSON parsing
                if rng.random() < 0.0002:
                    line = line[: max(1, len(line) // 2)]
                f.write(line + "\n")
        written += len(records)
        part += 1
        print(f"  [{out_subdir}] wrote part-{part:04d}.json ({len(records)} rows, {written}/{total_rows})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5_000_000)
    parser.add_argument("--clean-existing", action="store_true", default=True)
    args = parser.parse_args()

    if args.clean_existing and os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    n_core = int(args.rows * SOURCE_SPLIT["core_transactions"])
    n_mobile = int(args.rows * SOURCE_SPLIT["mobile_events"])
    n_legacy = args.rows - n_core - n_mobile

    rows_per_file = min(ROWS_PER_FILE, max(5000, args.rows // 10))

    print(f"Generating ~{args.rows:,} synthetic transactions into {OUT_DIR}")
    print(f"  core_transactions: {n_core:,} rows")
    write_csv_partitions(generate_core_chunk, n_core, "core_transactions", rows_per_file)

    print(f"  mobile_events: {n_mobile:,} rows")
    write_json_partitions(generate_mobile_chunk, n_mobile, "mobile_events", rows_per_file)

    print(f"  legacy_feed: {n_legacy:,} rows")
    write_csv_partitions(generate_legacy_chunk, n_legacy, "legacy_feed", rows_per_file)

    print("Done.")


if __name__ == "__main__":
    main()
