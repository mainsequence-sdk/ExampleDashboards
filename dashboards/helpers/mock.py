import mainsequence.client as msc
import mainsequence.instruments as msi
import pytz
from mainsequence.virtualfundbuilder.data_nodes import PortfolioFromDF, All_PORTFOLIO_COLUMNS, \
    WEIGHTS_TO_PORTFOLIO_COLUMNS
from mainsequence.virtualfundbuilder.portfolio_interface import PortfolioInterface
from mainsequence.tdag import DataNode, APIDataNode
from mainsequence.client.models_tdag import UpdateStatistics, ColumnMetaData

import numpy as np
import pandas as pd
import re
import json
import datetime
import QuantLib as ql
UTC = pytz.utc



# =========================================================
# 1) DRY helper: ensure both test assets exist and have pricing details
# =========================================================
def ensure_test_assets(unique_identifiers=None):
    """
    Ensure the two test bonds exist and have instrument pricing details.
    Returns: List[msc.Asset]
    """
    FLOATING_INDEX_NAME = "SOFR"


    if unique_identifiers is None:
        unique_identifiers = ["TEST_FLOATING_BOND_UST", "TEST_FIXED_BOND_USD"]

    # Fetch any existing
    existing_assets = msc.Asset.filter(unique_identifier__in=unique_identifiers)  # cookbook filtering
    uid_to_asset = {a.unique_identifier: a for a in existing_assets}

    # Build common dates (UTC)
    now_utc = datetime.datetime.now(UTC)
    time_idx = datetime.datetime(
        now_utc.year, now_utc.month, now_utc.day, now_utc.hour, now_utc.minute, tzinfo=UTC
    )

    # Common instrument kwargs
    common_kwargs = {
        "face_value": 100,
        "coupon_frequency": ql.Period(6, ql.Months),
        "day_count": ql.Actual365Fixed(),
        "calendar": ql.UnitedStates(ql.UnitedStates.GovernmentBond),
        "business_day_convention": ql.Unadjusted,
        "settlement_days": 0,
        "maturity_date": time_idx.date() + datetime.timedelta(days=365 * 10),
        "issue_date": time_idx.date(),
        "benchmark_rate_index_name":FLOATING_INDEX_NAME
    }

    created_assets = []
    for uid in unique_identifiers:
        asset = uid_to_asset.get(uid)
        needs_build = (asset is None) or (getattr(asset, "current_pricing_detail", None) is None)

        if needs_build:
            # Build instrument
            if "FLOATING" in uid:
                instrument = msi.FloatingRateBond(
                    **common_kwargs,
                    floating_rate_index_name=FLOATING_INDEX_NAME,
                )
            else:
                instrument = msi.FixedRateBond(
                    **common_kwargs,
                    coupon_rate=0.05,
                )

            # Minimal registration payload for a custom asset (keeps your approach)
            payload_item = {
                "unique_identifier": uid,
                "snapshot": {"name": uid, "ticker": uid},
            }
            # Your environment already uses this utility; keep it DRY.
            registered = msc.Asset.batch_get_or_register_custom_assets([payload_item])
            asset = registered[0]

            # Attach instrument pricing details
            asset.add_instrument_pricing_details_from_ms_instrument(
                instrument=instrument, pricing_details_date=time_idx
            )

        created_assets.append(asset)

    return created_assets


# =========================================================
# 2) DataNode: daily close simulation (MultiIndex: time_index, unique_identifier)
# =========================================================
class SimulatedDailyClosePrices(DataNode):
    """
    Simulates daily 'close' prices for a list of assets.
    Output index: MultiIndex ('time_index', 'unique_identifier'), UTC.
    Columns: 'close' (float, lowercase <=63 chars). No datetime columns.
    """

    _ARGS_IGNORE_IN_STORAGE_HASH = ["asset_list"]  # asset universe shouldn't change the storage identity

    def __init__(self, asset_list=None, *args, **kwargs):
        self.asset_list = asset_list or []
        super().__init__(*args, **kwargs)

    def dependencies(self) -> dict[str, "DataNode | APIDataNode"]:
        return {}

    # Optional: supply the asset universe to the platform if not provided at ctor
    def get_asset_list(self):
        return self.asset_list or ensure_test_assets()

    def get_column_metadata(self):
        return [ColumnMetaData(
            column_name="close",
            dtype="float",
            label="Close",
            description="Simulated daily close price"
        )]

    def get_table_metadata(self):
        return msc.TableMetaData(
            identifier="simulated_daily_closes_f1_v1",
            data_frequency_id=msc.DataFrequency.one_d,
            description="Daily close prices simulated via lognormal steps."
        )

    # ---- Helper to determine last price across prior observations ----
    @staticmethod
    def _last_close_for(asset_uid: str, obs_df: pd.DataFrame, fallback: float = 100.0) -> float:
        if obs_df is None or obs_df.empty:
            return fallback
        try:
            ser = (
                obs_df.reset_index()
                .sort_values(["unique_identifier", "time_index"])
                .set_index(["time_index", "unique_identifier"])["close"]
            )
            last = ser.xs(asset_uid, level="unique_identifier").dropna()
            return float(last.iloc[-1]) if len(last) else fallback
        except Exception:
            return fallback

    def update(self) -> pd.DataFrame:
        """
        Incremental pattern:
        - prior observations fetched once using get_update_range_map_great_or_equal()
        - per-asset start = last update + 1 day
        - end = yesterday 00:00 UTC
        - generate deterministic lognormal path (seeded by uid) for idempotency
        """
        us: UpdateStatistics = self.update_statistics

        # One-shot fetch of prior observations
        range_descriptor = us.get_update_range_map_great_or_equal()
        prior_obs = self.get_ranged_data_per_asset(range_descriptor=range_descriptor)

        # Target end = yesterday 00:00 UTC
        yday = (datetime.datetime.now(UTC)
                .replace(hour=0, minute=0, second=0, microsecond=0)
                - datetime.timedelta(days=1))

        frames = []
        for asset in us.asset_list:
            # Start = last + 1 day at 00:00
            start_time = us.get_asset_earliest_multiindex_update(asset) + datetime.timedelta(days=1)
            start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC)

            if start_time > yday:
                continue

            idx = pd.date_range(start=start_time, end=yday, freq="D", tz=UTC, name="time_index")
            if len(idx) == 0:
                continue

            seed_price = self._last_close_for(asset.unique_identifier, prior_obs, fallback=100.0)

            # Stable randomness per asset for idempotency
            rng = np.random.default_rng(abs(hash(asset.unique_identifier)) % (2**32))
            steps = rng.lognormal(mean=0.0, sigma=0.01/np.sqrt(252), size=len(idx))
            path = seed_price * np.cumprod(steps)

            tmp = pd.DataFrame({asset.unique_identifier: path}, index=idx)
            frames.append(tmp)

        if not frames:
            return pd.DataFrame()

        wide = pd.concat(frames, axis=1)                        # columns = unique_identifiers
        long = wide.melt(ignore_index=False, var_name="unique_identifier", value_name="close")
        long = long.set_index("unique_identifier", append=True) # -> (time_index, unique_identifier)
        return long


class TestFixedIncomePortfolio(PortfolioFromDF):
    def get_portfolio_df(self):

        time_idx = datetime.datetime.now()
        time_idx = datetime.datetime(time_idx.year, time_idx.month, time_idx.day, time_idx.hour,
                                     time_idx.minute, tzinfo=pytz.utc, )

        assets = ensure_test_assets()
        unique_identifiers = [a.unique_identifier for a in assets]

        # ----- build dict-valued columns -----
        keys = unique_identifiers
        n = len(keys)

        # random weights that sum to 1
        import numpy as np
        w = np.random.rand(n)
        w = w / w.sum()
        weights_dict = json.dumps({k: float(v) for k, v in zip(keys, w)})

        # everything else set to 1 per asset
        ones_dict = json.dumps({k: 1 for k in keys})



        # Map logical fields to actual DF columns
        col_weights_current = "rebalance_weights"  # "weights_current"
        col_price_current = "rebalance_price"  # "price_current"
        col_vol_current = "volume"  # "volume_current"
        col_weights_before = "weights_at_last_rebalance"  # "weights_before"
        col_price_before = "price_at_last_rebalance"  # "price_before"
        col_vol_before = "volume_at_last_rebalance"  # "volume_before"

        row = {
            "time_index": time_idx,
            "close": 1,
            "return": 0,
            "last_rebalance_date": time_idx.timestamp(),
            col_weights_current: weights_dict,
            col_weights_before: weights_dict,  # same as current
            col_price_current: ones_dict,
            col_price_before: ones_dict,
            col_vol_current: ones_dict,
            col_vol_before: ones_dict,
        }

        # one-row DataFrame
        portoflio_df = pd.DataFrame([row])
        portoflio_df = portoflio_df.set_index("time_index")
        if self.update_statistics.max_time_index_value is not None:
            portoflio_df = portoflio_df[portoflio_df.index > self.update_statistics.max_time_index_value]
        return portoflio_df

def build_test_portfolio(portfolio_name:str):
    node = TestFixedIncomePortfolio(portfolio_name=portfolio_name, calendar_name="24/7",
                         target_portfolio_about="Test")

    PortfolioInterface.build_and_run_portfolio_from_df(portfolio_node=node,
                                                       add_portfolio_to_markets_backend=True)

    #prices also need to be run to have a simulated impact
    assets = ensure_test_assets()

    # Instantiate and update the DataNode (platform would orchestrate this)
    prices_node = SimulatedDailyClosePrices(asset_list=assets)
    prices_node.run(debug_mode=True, force_update=True)



