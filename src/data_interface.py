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


class APITimeSeries:
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
            asof = datetime.date(2025, 8, 11)

            return {
                "asof": asof,
                "zeros": [
                    {"tenor": "1M", "yield": 0.0370},
                    {"tenor": "3M", "yield": 0.0385},
                    {"tenor": "6M", "yield": 0.0395},
                    {"tenor": "9M", "yield": 0.0405},
                    {"tenor": "1Y", "yield": 0.0410},
                ],
                "fixed_rate_bonds": [
                    # clean/dirty per 100 face; semiannual coupons by default
                    {"tenor": "2Y", "coupon": 0.0425, "clean_price": 99.20, "dirty_price": 99.45, "frequency": "6M"},
                    {"tenor": "3Y", "coupon": 0.0440, "clean_price": 98.85, "dirty_price": 99.10, "frequency": "6M"},
                    {"tenor": "5Y", "coupon": 0.0475, "clean_price": 98.10, "dirty_price": 98.40, "frequency": "6M"},
                    {"tenor": "7Y", "coupon": 0.0490, "clean_price": 97.25, "dirty_price": 97.60, "frequency": "6M"},
                    {"tenor": "10Y", "coupon": 0.0500, "clean_price": 96.80, "dirty_price": 97.20, "frequency": "6M"},
                ]
            }
        else:
            raise ValueError(f"Table '{table_name}' not found in mock data API.")

