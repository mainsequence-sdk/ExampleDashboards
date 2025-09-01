import datetime
from typing import Dict, Optional, TypedDict, Any
import random
from src.utils import to_ql_date
import QuantLib as ql


class DateInfo(TypedDict, total=False):
    """Defines the date range for a data query."""
    start_date: Optional[datetime.datetime]
    start_date_operand: Optional[str]
    end_date: Optional[datetime.datetime]
    end_date_operand: Optional[str]


UniqueIdentifierRangeMap = Dict[str, DateInfo]


class APIDataNode:
    """
    A mock class to simulate fetching financial time series data from an API.

    In a real-world scenario, this class would contain logic to connect to a
    financial data provider (e.g., Bloomberg, Refinitiv, a database).
    """

    @staticmethod
    def get_historical_fixings(index_name: str, start_date: datetime.date, end_date: datetime.date) -> Dict[
        datetime.date, float]:
        """
        Simulates fetching historical index fixings from a database.

        CORRECTED: This now dynamically selects the appropriate calendar based on the index name.
        """
        print(f"--- MOCK DATA API ---")
        print(f"Fetching historical fixings for '{index_name}' from {start_date} to {end_date}")

        # Dynamically select the calendar based on the index name
        calendar = ql.TARGET()  # Default calendar
        if 'USDLibor' in index_name:
            calendar = ql.UnitedKingdom()
            print("Using UnitedKingdom calendar for LIBOR.")
        elif 'Euribor' in index_name:
            calendar = ql.TARGET()  # TARGET is the standard for EUR rates
            print("Using TARGET calendar for Euribor.")
        elif 'SOFR' in index_name:
            calendar = ql.UnitedStates(ql.UnitedStates.SOFR)
            print("Using UnitedStates.SOFR calendar for SOFR.")
        elif 'TIIE' in index_name or 'F-TIIE' in index_name:
            # Mexico interbank conventions
            try:
                calendar = ql.Mexico()
                print("Using Mexico calendar for TIIE.")
            except Exception:
                calendar = ql.TARGET()
                print("Mexico calendar unavailable in this wheel; falling back to TARGET.")

        print("---------------------\n")

        fixings = {}
        current_date = start_date
        base_rate = 0.05

        while current_date <= end_date:
            ql_date = to_ql_date(current_date)
            # Only generate a fixing if the date is a business day for the selected calendar
            if calendar.isBusinessDay(ql_date):
                random_factor = (random.random() - 0.5) * 0.001
                fixings[current_date] = base_rate + random_factor

            current_date += datetime.timedelta(days=1)

        return fixings

    @staticmethod
    def get_historical_data(table_name: str, asset_range_map: UniqueIdentifierRangeMap) -> Dict[str, Any]:
        """
        Simulates fetching historical data for a given asset or data type.

        Args:
            table_name: The name of the data table to query.
            asset_range_map: A dictionary mapping identifiers to date ranges.

        Returns:
            A dictionary containing mock market data.
        """
        print(f"--- MOCK DATA API ---")
        print(f"Fetching data from table '{table_name}' for assets: {list(asset_range_map.keys())}")
        print("---------------------\n")

        if table_name == "equities_daily":
            asset_ticker = list(asset_range_map.keys())[0]
            mock_data = {
                asset_ticker: {
                    "spot_price": 175.50,
                    "volatility": 0.20,
                    "dividend_yield": 0.015,
                    "risk_free_rate": 0.04
                }
            }
            if asset_ticker in mock_data:
                return mock_data[asset_ticker]
            else:
                raise ValueError(f"No mock data available for asset: {asset_ticker}")

        elif table_name == "interest_rate_swaps":
            # A more realistic set of market rates for curve bootstrapping.
            # This includes short-term deposit rates and longer-term swap rates.
            return {
                "curve_nodes": [
                    {'type': 'deposit', 'tenor': '3M', 'rate': 0.048},
                    {'type': 'deposit', 'tenor': '6M', 'rate': 0.050},
                    {'type': 'swap', 'tenor': '1Y', 'rate': 0.052},
                    {'type': 'swap', 'tenor': '2Y', 'rate': 0.054},
                    {'type': 'swap', 'tenor': '3Y', 'rate': 0.055},
                    {'type': 'swap', 'tenor': '5Y', 'rate': 0.056},
                    {'type': 'swap', 'tenor': '10Y', 'rate': 0.057},
                ]
            }
        elif table_name == "discount_bond_curve":
            # Zero rates for discounting bond cashflows (simple upward-sloping curve).
            # Tenors are parsed by QuantLib (e.g., "6M", "5Y").
            return {
                "curve_nodes": [
                    # --- Zero-coupon section (<= 1Y) ---
                    {"type": "zcb", "days_to_maturity": 30, "yield": 0.0370},
                    {"type": "zcb", "days_to_maturity": 90, "yield": 0.0385},
                    {"type": "zcb", "days_to_maturity": 180, "yield": 0.0395},
                    {"type": "zcb", "days_to_maturity": 270, "yield": 0.0405},
                    {"type": "zcb", "days_to_maturity": 360, "yield": 0.0410},

                    # --- Coupon bond section (>= 2Y) ---
                    {"type": "bond", "days_to_maturity": 730, "coupon": 0.0425, "clean_price": 99.20,
                     "dirty_price": 99.45, "frequency": "6M", "day_count": "30/360"},
                    {"type": "bond", "days_to_maturity": 1095, "coupon": 0.0440, "clean_price": 98.85,
                     "dirty_price": 99.10, "frequency": "6M", "day_count": "30/360"},
                    {"type": "bond", "days_to_maturity": 1825, "coupon": 0.0475, "clean_price": 98.10,
                     "dirty_price": 98.40, "frequency": "6M", "day_count": "30/360"},
                    {"type": "bond", "days_to_maturity": 2555, "coupon": 0.0490, "clean_price": 97.25,
                     "dirty_price": 97.60, "frequency": "6M", "day_count": "30/360"},
                    {"type": "bond", "days_to_maturity": 3650, "coupon": 0.0500, "clean_price": 96.80,
                     "dirty_price": 97.20, "frequency": "6M", "day_count": "30/360"},
                ]
            }
        elif table_name == "fx_options":
            # Mock FX options market data
            currency_pair = list(asset_range_map.keys())[0]

            # Mock data for common currency pairs
            fx_mock_data = {
                "EURUSD": {
                    "spot_fx_rate": 1.0850,
                    "volatility": 0.12,
                    "domestic_rate": 0.045,  # USD rate
                    "foreign_rate": 0.035    # EUR rate
                },
                "GBPUSD": {
                    "spot_fx_rate": 1.2650,
                    "volatility": 0.15,
                    "domestic_rate": 0.045,  # USD rate
                    "foreign_rate": 0.040    # GBP rate
                },
                "USDJPY": {
                    "spot_fx_rate": 148.50,
                    "volatility": 0.11,
                    "domestic_rate": 0.005,  # JPY rate
                    "foreign_rate": 0.045    # USD rate
                },
                "USDCHF": {
                    "spot_fx_rate": 0.8950,
                    "volatility": 0.13,
                    "domestic_rate": 0.015,  # CHF rate
                    "foreign_rate": 0.045    # USD rate
                }
            }

            if currency_pair in fx_mock_data:
                return fx_mock_data[currency_pair]
            else:
                # Default mock data for unknown pairs
                return {
                    "spot_fx_rate": 1.0000,
                    "volatility": 0.15,
                    "domestic_rate": 0.040,
                    "foreign_rate": 0.040
                }

        elif table_name == "tiie_zero_valmer":
            """
            Return a pre-built MXN TIIE zero curve parsed from a CSV.

            Expected CSV columns (case-insensitive; flexible):
              - Either 'maturity_date' (YYYY-MM-DD) OR 'days_to_maturity' OR a 'tenor' like '28D','3M','2Y'
              - One rate column among: ['zero','rate','yield','tiie'] as a decimal (e.g., 0.095 for 9.5%)
                (if the file holds percents like 9.50, we'll auto-convert to 0.095)
            """
            import os
            import pandas as pd
            from pathlib import Path

            # You can override this path in your env; default points to the uploaded file
            DEFAULT_TIIE_CSV = Path(__file__).resolve().parents[2] / "data" / "MEXDERSWAP_IRSTIIEPR.csv"
            csv_path = os.getenv("TIIE_ZERO_CSV") or str(DEFAULT_TIIE_CSV)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"TIIE zero curve CSV not found at: {csv_path}")

            names = ["id", "curve_name", "maturity_date", "days_to_maturity", "zero_rate"]
            # STRICT: comma-separated, headerless, exactly these six columns
            df = pd.read_csv(csv_path, header=None, names=names, sep=",", engine="c", dtype=str)
            # pick a rate column

            df["days_to_maturity"]=df["days_to_maturity"].astype(int)
            df["zero_rate"] = df["zero_rate"].astype(float)/100

            nodes = [
                {"days_to_maturity": d, "zero": z}
                for d, z in zip(df["days_to_maturity"], df["zero_rate"])
                if d > 0
            ]
            return {"curve_nodes": nodes}

        else:
            raise ValueError(f"Table '{table_name}' not found in mock data API.")
