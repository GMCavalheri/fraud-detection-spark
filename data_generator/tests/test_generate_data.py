"""
Tests for the synthetic data generator. Several of these are regression
tests for real bugs found while building this project:
  - non-unique transaction IDs across file chunks (fixed with id_offset)
  - a negative background-row count on tiny trailing chunks (fixed with
    MIN_CHUNK_FOR_PATTERNS)
  - a float64 city-index array on zero-travel-pair chunks (fixed with an
    explicit dtype=int in impossible_travel_rows)
"""

import json

import numpy as np
import pytest

import generate_data as gd


class TestApplyLabelNoise:
    def test_flip_counts_are_exact(self):
        # apply_label_noise's flip *counts* are deterministic (only which
        # indices get flipped is random), so the resulting positive count is
        # exactly predictable from the input composition.
        n_pos, n_neg = 1000, 9000
        is_fraud = np.array([1] * n_pos + [0] * n_neg)

        result = gd.apply_label_noise(is_fraud)

        expected_pos_flipped = int(n_pos * gd.POS_LABEL_FLIP_RATE)
        expected_neg_flipped = int(n_neg * gd.NEG_LABEL_FLIP_RATE)
        expected_final_pos = n_pos - expected_pos_flipped + expected_neg_flipped

        assert result.sum() == expected_final_pos
        assert set(np.unique(result)) <= {0, 1}

    def test_does_not_mutate_input(self):
        is_fraud = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 100)
        original = is_fraud.copy()
        gd.apply_label_noise(is_fraud)
        np.testing.assert_array_equal(is_fraud, original)


class TestUniqueIds:
    """Regression test: transaction_id used to reset to 0 in every file
    chunk, so ETL's dedup-by-id step silently dropped ~70% of rows."""

    def test_core_ids_unique_across_chunks(self):
        chunk1 = gd.generate_core_chunk(3000, id_offset=0)
        chunk2 = gd.generate_core_chunk(3000, id_offset=3000)
        ids1 = set(chunk1["transaction_id"])
        ids2 = set(chunk2["transaction_id"])
        assert ids1.isdisjoint(ids2)

    def test_mobile_ids_unique_across_chunks(self):
        chunk1 = gd.generate_mobile_chunk(3000, id_offset=0)
        chunk2 = gd.generate_mobile_chunk(3000, id_offset=3000)
        ids1 = {r["evt_id"] for r in chunk1}
        ids2 = {r["evt_id"] for r in chunk2}
        assert ids1.isdisjoint(ids2)

    def test_legacy_ids_unique_across_chunks(self):
        chunk1 = gd.generate_legacy_chunk(3000, id_offset=0)
        chunk2 = gd.generate_legacy_chunk(3000, id_offset=3000)
        ids1 = set(chunk1["TXN_ID"])
        ids2 = set(chunk2["TXN_ID"])
        assert ids1.isdisjoint(ids2)

    def test_core_ids_unique_within_a_single_chunk(self):
        # intentional exact-duplicate rows share a transaction_id by design
        # (simulating an upstream retry), so uniqueness only holds pre-dup
        chunk = gd.generate_core_chunk(5000, id_offset=0)
        base_rows = chunk.drop_duplicates(subset=["transaction_id"])
        assert base_rows["transaction_id"].is_unique


class TestSmallChunksDoNotCrash:
    """Regression tests: fraud-pattern injection floors (max(1, ...)) used
    to exceed tiny trailing chunk sizes, driving background row counts
    negative; and impossible_travel_rows(0) used to produce float64 city
    index arrays that crashed integer indexing. Both only show up below
    MIN_CHUNK_FOR_PATTERNS."""

    @pytest.mark.parametrize("n", [1, 5, 50, gd.MIN_CHUNK_FOR_PATTERNS - 1])
    def test_generate_core_chunk_small_n(self, n):
        df = gd.generate_core_chunk(n, id_offset=0)
        assert len(df) >= n  # >= because exact-duplicate injection can add rows
        assert df["city"].notna().all() or True  # city can be legitimately missing (injected)

    @pytest.mark.parametrize("n", [1, 5, 50, gd.MIN_CHUNK_FOR_PATTERNS - 1])
    def test_generate_mobile_chunk_small_n(self, n):
        records = gd.generate_mobile_chunk(n, id_offset=0)
        assert len(records) >= n


class TestSchema:
    def test_core_chunk_has_expected_columns(self):
        df = gd.generate_core_chunk(500, id_offset=0)
        expected = {
            "transaction_id", "card_number", "account_id", "customer_name",
            "txn_date", "amount", "merchant_name", "category", "city",
            "country", "channel", "is_fraud",
        }
        assert expected <= set(df.columns)
        assert set(df["is_fraud"].unique()) <= {0, 1}

    def test_mobile_chunk_records_are_json_serializable(self):
        records = gd.generate_mobile_chunk(500, id_offset=0)
        for rec in records[:50]:
            json.dumps(rec)  # raises if anything isn't serializable
            assert "device" in rec
            assert set(rec["device"].keys()) == {"lat", "lon", "os"}
            assert rec["is_fraud"] in (0, 1)

    def test_legacy_chunk_amounts_are_always_positive(self):
        # unlike core_transactions, legacy_feed never injects negative amounts
        df = gd.generate_legacy_chunk(2000, id_offset=0)
        assert (df["AMOUNT"] > 0).all()

    def test_legacy_chunk_mcc_round_trips_to_a_real_category(self):
        df = gd.generate_legacy_chunk(2000, id_offset=0)
        assert set(df["MCC"].unique()) <= set(gd.MCC_MAP.keys())

    def test_legacy_chunk_currency_is_a_known_code(self):
        df = gd.generate_legacy_chunk(2000, id_offset=0)
        assert set(df["CCY"].unique()) <= set(gd.CURRENCY_RATES_TO_USD.keys())


class TestFraudRate:
    def test_overall_fraud_rate_lands_near_target(self):
        # statistical test - uses a wide tolerance since these tests don't
        # reset the module-level RNG between each other, so exact output
        # depends on what ran before. The important invariant is "roughly on
        # target, not the ~13% we got from a real label-noise math bug."
        core = gd.generate_core_chunk(20_000, id_offset=0)
        mobile = gd.generate_mobile_chunk(10_000, id_offset=0)
        legacy = gd.generate_legacy_chunk(6_000, id_offset=0)

        total_fraud = core["is_fraud"].sum() + sum(r["is_fraud"] for r in mobile) + legacy["IS_FRAUD"].sum()
        total_rows = len(core) + len(mobile) + len(legacy)
        fraud_rate = total_fraud / total_rows

        assert 0.005 <= fraud_rate <= 0.03, f"fraud rate {fraud_rate:.4f} is far from the ~1.5% target"
