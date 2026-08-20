import os

import etl_clean as ec


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


class TestReadCore:
    def test_parses_all_three_date_formats_and_cleans_amount(self, spark, tmp_path):
        raw_dir = str(tmp_path)
        _write(
            os.path.join(raw_dir, "core_transactions", "part-0000.csv"),
            "transaction_id,card_number,account_id,customer_name,txn_date,amount,merchant_name,category,city,country,channel,is_fraud\n"
            "CORE-000000001,1234567890123456,ACCT-1,Alice,2026-01-15T10:23:00,$1234.56,Acme Co,electronics,London,UK,online,0\n"
            'CORE-000000002,1234567890123457,ACCT-2,Bob,01/15/2026 10:23 AM,"1,234.56",Acme Co,ELECTRONICS,Paris,France,pos,0\n'
            "CORE-000000003,1234567890123458,ACCT-3,Carol,1768472580,-45.00,Acme Co,Electronics,Berlin,Germany,online,1\n",
        )

        out, raw_count = ec.read_core(spark, raw_dir=raw_dir)
        rows = {r["transaction_id"]: r for r in out.collect()}

        assert raw_count == 3
        # all three date formats parse to a non-null timestamp
        assert all(rows[f"CORE-00000000{i}"]["event_ts"] is not None for i in (1, 2, 3))
        # category normalized to title case regardless of input casing
        assert rows["CORE-000000001"]["category"] == "Electronics"
        assert rows["CORE-000000002"]["category"] == "Electronics"
        # dirty amount strings ($, commas) parsed and abs()'d into amount_usd
        assert rows["CORE-000000001"]["amount_usd"] == 1234.56
        assert rows["CORE-000000002"]["amount_usd"] == 1234.56
        # negative amount flagged and corrected to positive
        assert rows["CORE-000000003"]["amount_usd"] == 45.00
        assert rows["CORE-000000003"]["_had_negative_amount"] is True

    def test_card_number_is_masked_not_carried_through(self, spark, tmp_path):
        raw_dir = str(tmp_path)
        _write(
            os.path.join(raw_dir, "core_transactions", "part-0000.csv"),
            "transaction_id,card_number,account_id,customer_name,txn_date,amount,merchant_name,category,city,country,channel,is_fraud\n"
            "CORE-000000001,4111111111111234,ACCT-1,Alice,2026-01-15T10:23:00,10.00,Acme,Fuel,London,UK,pos,0\n",
        )
        out, _ = ec.read_core(spark, raw_dir=raw_dir)
        row = out.collect()[0]
        assert "card_number" not in out.columns
        assert row["card_last4"] == "1234"
        assert len(row["card_hash"]) == 12


class TestReadMobile:
    def test_currency_and_missing_merchant_and_corrupt_lines(self, spark, tmp_path):
        raw_dir = str(tmp_path)
        good1 = '{"evt_id":"MOB-1","acct_card":"ACCT-1","event_ts":"2026-01-15T10:23:00+0000","amt":50.0,"amt_currency":"usd","merchant":"Shop","merchant_category":"Groceries","device":{"lat":1.0,"lon":2.0,"os":"iOS"},"city_name":"Dubai","is_fraud":"0"}'
        good2 = '{"evt_id":"MOB-2","acct_card":"ACCT-2","event_ts":"2026-01-15T10:23:00+0000","amt":100.0,"amt_currency":"EUR","merchant":null,"merchant_category":"Travel","device":{"lat":3.0,"lon":4.0,"os":"Android"},"city_name":"Paris","is_fraud":"1"}'
        corrupt = good2[: len(good2) // 2]
        _write(
            os.path.join(raw_dir, "mobile_events", "part-0000.json"),
            "\n".join([good1, good2, corrupt]) + "\n",
        )

        out, total_lines, corrupt_lines = ec.read_mobile(spark, raw_dir=raw_dir)
        rows = {r["transaction_id"]: r for r in out.collect()}

        assert total_lines == 3
        assert corrupt_lines == 1
        assert rows["MOB-1"]["currency_original"] == "USD"
        assert rows["MOB-1"]["amount_usd"] == 50.0  # USD passes through unchanged
        assert rows["MOB-2"]["amount_usd"] != 100.0  # EUR gets converted
        assert rows["MOB-2"]["merchant"] is None


class TestReadLegacy:
    def test_currency_conversion_and_mcc_mapping(self, spark, tmp_path):
        raw_dir = str(tmp_path)
        _write(
            os.path.join(raw_dir, "legacy_feed", "part-0000.csv"),
            "TXN_ID,ACCOUNT_REF,CARD_DISPLAY,CUST_NAME,DATE,AMOUNT,CCY,MERCHANT,MCC,COUNTRY,IS_FRAUD\n"
            "LEG-000000001,ACCT-1,****1234,Dave,15-Jan-2026 10:23,100.00,USD,Shop,5411,USA,0\n"
            "LEG-000000002,ACCT-2,****5678,Erin,15-Jan-2026 10:23,100.00,EUR,Shop,5812,France,0\n",
        )
        out, raw_count = ec.read_legacy(spark, raw_dir=raw_dir)
        rows = {r["transaction_id"]: r for r in out.collect()}

        assert raw_count == 2
        assert rows["LEG-000000001"]["category"] == "Groceries"
        assert rows["LEG-000000002"]["category"] == "Restaurants"
        # 100 EUR should convert to more USD than 100 USD (EUR rate > 1.0)
        assert rows["LEG-000000002"]["amount_usd"] > rows["LEG-000000001"]["amount_usd"]
        assert rows["LEG-000000001"]["card_last4"] == "1234"


class TestGeocodeMissingCoords:
    def test_fills_lat_lon_from_city_when_missing(self, spark):
        df = spark.createDataFrame(
            [("New York", None, None), ("Unknown City", None, None), ("London", 1.0, 1.0)],
            ["city", "lat", "lon"],
        )
        out = {r["city"]: r for r in ec.geocode_missing_coords(df).collect()}

        assert out["New York"]["lat"] == 40.7128
        assert out["Unknown City"]["lat"] is None  # not in the lookup - stays null
        assert out["London"]["lat"] == 1.0  # existing coords are never overwritten
