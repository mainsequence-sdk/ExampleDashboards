# dashboards/markets/positions.py
from __future__ import annotations

import datetime as dt
from typing import Dict, Iterable, Mapping, Optional, Tuple, Any, List, Callable, TypeVar, Type

import pandas as pd

import QuantLib as ql  # evaluation date handling lives elsewhere; here we just rely on instruments' curves
from dashboards.services.curves import zspread_from_dirty_ccy
from mainsequence.instruments.instruments.position import Position, PositionLine
import mainsequence.instruments as msi


class PositionOperations:
    """
    Operations that act on *positions only* (no portfolio, no UI):
      - instantiate base/bumped positions from a position template + curve maps
      - compute and propagate z-spreads from dirty prices
      - compute per-line/base/bumped NPVs
      - carry-to-cutoff and per-line NPV tables
      - local single-line bump (for details views)

    Notes
    -----
    • This class **does not** fetch curves or signals and **does not** talk to Streamlit.
    • Curve building and portfolio concerns remain elsewhere (e.g., in your context builder and PortfoliosOperations).
    """

    def __init__(self, position_template: Optional[Position] = None):
        # Template from which base/bumped positions can be instantiated
        self._template: Optional[Position] = position_template

        # Last instantiated positions (optional)
        self.base_position: Optional[Position] = None
        self.bumped_position: Optional[Position] = None

        # Curve maps kept for reference (optional)
        self.base_curves_by_index: Dict[str, ql.YieldTermStructureHandle] = {}
        self.bumped_curves_by_index: Dict[str, ql.YieldTermStructureHandle] = {}

        # Valuation date (optional; QL Settings is assumed to be set upstream)
        self.valuation_date: Optional[dt.date] = None

    @classmethod
    def from_position(cls,position:Position,base_curves_by_index: Dict[str, ql.YieldTermStructureHandle],
                      valuation_date:dt.date,
                      ):
        po=cls(position_template=position)
        po.base_curves_by_index=base_curves_by_index
        po.valuation_date=valuation_date
        return po

    def _instantiate_from_template(
            self,
            template: Position,
            *,
            index_curve_map: Mapping[str, ql.YieldTermStructureHandle],
            valuation_date: dt.date,
    ) -> Position:
        """
        Pure helper: deep‑copy template instruments, set valuation_date,
        and reset each curve from `index_curve_map` keyed by
        instrument.floating_rate_index_name. Copies extra_market_info.
        """
        new_lines = []
        for line in template.lines:
            inst = line.instrument.copy()
            inst.set_valuation_date(valuation_date)
            if isinstance(inst, msi.FloatingRateBond):
                idx_name = getattr(inst, "floating_rate_index_name", None)
                if idx_name is None:
                    raise KeyError(f"No curve provided for index '{idx_name}' during instantiation.")

                if idx_name not in index_curve_map:
                    raise KeyError(f"No curve provided for index '{idx_name}' during instantiation.")
            elif isinstance(inst, msi.FixedRateBond):
                idx_name = getattr(inst, "benchmark_rate_index_name", None)
                if idx_name is None:
                    raise  Exception(f"{inst} needs a benchmark_rate_index_name ")
            else:
                raise Exception(f"{inst} is not a bond")
            inst.reset_curve(index_curve_map[idx_name])

            xmi = dict(getattr(line, "extra_market_info", {}) or {})
            new_lines.append(
                PositionLine(units=line.units, instrument=inst, extra_market_info=xmi)
            )
        return Position(lines=new_lines)

    # -------------------------------------------------------------------------
    # Template & curves
    # -------------------------------------------------------------------------
    def set_template(self, template_position: Position) -> None:
        """Set or replace the position template."""
        self._template = template_position

    def set_curves(
        self,
        *,
        base_curves_by_index: Optional[Mapping[str, ql.YieldTermStructureHandle]] = None,
        bumped_curves_by_index: Optional[Mapping[str, ql.YieldTermStructureHandle]] = None,
    ) -> None:
        """Record curve handles (no side effects on positions)."""
        if base_curves_by_index is not None:
            self.base_curves_by_index = dict(base_curves_by_index)
        if bumped_curves_by_index is not None:
            self.bumped_curves_by_index = dict(bumped_curves_by_index)

    def set_valuation_date(self, d: dt.date) -> None:
        self.valuation_date = d

    # -------------------------------------------------------------------------
    # Instantiation (from template) — prefers your existing instantiate_position
    # -------------------------------------------------------------------------
    def instantiate_base(self) -> Position:
        """
        Instantiate the base position from the current template and base_curves_by_index.
        """
        if self._template is None:
            raise RuntimeError("PositionOperations.instantiate_base: template is not set.")
        if not self.base_curves_by_index:
            raise RuntimeError("PositionOperations.instantiate_base: base_curves_by_index is empty.")
        if self.valuation_date is None:
            raise RuntimeError("PositionOperations.instantiate_base: valuation_date is not set.")

        self.base_position = self._instantiate_from_template(
            self._template,
            index_curve_map=self.base_curves_by_index,
            valuation_date=self.valuation_date,
        )
        return self.base_position

    def instantiate_bumped(self) -> Position:
        """
        Instantiate the bumped position from the current template and bumped_curves_by_index.
        """
        if self._template is None:
            raise RuntimeError("PositionOperations.instantiate_bumped: template is not set.")
        if not self.bumped_curves_by_index:
            raise RuntimeError("PositionOperations.instantiate_bumped: bumped_curves_by_index is empty.")
        if self.valuation_date is None:
            raise RuntimeError("PositionOperations.instantiate_bumped: valuation_date is not set.")

        self.bumped_position = self._instantiate_from_template(
            self._template,
            index_curve_map=self.bumped_curves_by_index,
            valuation_date=self.valuation_date,
        )
        return self.bumped_position

    # -------------------------------------------------------------------------
    # z-spreads from dirty price (base), mirrored to bumped
    # -------------------------------------------------------------------------
    def compute_and_apply_z_spreads_from_dirty_price(
        self,
        *,
        base_position: Optional[Position] = None,
        bumped_position: Optional[Position] = None,
    ) -> None:
        """
        Compute z-spread per line to match dirty_price in extra_market_info on the *base* position,
        then mirror the same z-spread onto the corresponding line in the bumped position.

        This mirrors the logic you already use in `context.build_context`.
        """
        pos_b = base_position or self.base_position
        pos_u = bumped_position or self.bumped_position

        if pos_b is None:
            raise RuntimeError("compute_and_apply_z_spreads_from_dirty_price: base_position is required.")

        # If both are provided and were instantiated from the same template, lines align by order.
        iterator: Iterable[Tuple[Any, Optional[Any]]]
        if pos_u is not None and len(pos_b.lines) == len(pos_u.lines):
            iterator = zip(pos_b.lines, pos_u.lines)
        else:
            iterator = ((ln, None) for ln in pos_b.lines)

        for line_b, line_u in iterator:
            dirty_price = (line_b.extra_market_info or {}).get("dirty_price", None)
            if dirty_price is None:
                # Keep consistent with your context: fail visibly or set z=0; you had a try/except -> z=0.
                # Here we choose to fail fast to catch misconfigurations early.
                raise ValueError(f"Line has no 'dirty_price' in extra_market_info: {line_b}")

            # Ensure the instrument engine is built
            line_b.instrument.price()
            try:
                # Use the curve assigned to the instrument as the discount curve
                base_curve_for_line = line_b.instrument.get_index_curve()
                z = zspread_from_dirty_ccy(line_b.instrument._bond, float(dirty_price), discount_curve=base_curve_for_line)
            except Exception:
                # Keep behavior compatible with your context fallback
                z = 0.0

            line_b.extra_market_info["z_spread"] = z
            if line_u is not None:
                line_u.extra_market_info["z_spread"] = z

    # -------------------------------------------------------------------------
    # Per-line local bump (no full bumped position required)
    # -------------------------------------------------------------------------
    def bumped_price_for_line(
        self,
        line,
        *,
        bumped_curves_by_index: Optional[Mapping[str, ql.YieldTermStructureHandle]] = None,
    ) -> float:
        """
        Compute this line's *bumped* price by copying the instrument and resetting its curve
        to the curve for its floating rate index, falling back to the instrument's existing curve
        if the index is not found. Mirrors the logic you use in `asset_detail`.
        """
        inst = line.instrument.copy()
        idx_name = getattr(line.instrument, "floating_rate_index_name", None)
        curve_map = dict(bumped_curves_by_index or self.bumped_curves_by_index or {})
        bump_curve = curve_map.get(idx_name, None)
        if bump_curve is None:
            bump_curve = line.instrument.get_index_curve()
        inst.reset_curve(bump_curve)
        return float(inst.price())

    # -------------------------------------------------------------------------
    # NPVs & carry
    # -------------------------------------------------------------------------
    @staticmethod
    def position_total_npv(position: Position) -> float:
        return float(sum(float(ln.units) * float(ln.instrument.price()) for ln in position.lines))

    @staticmethod
    def _agg_net_cashflows(position: Position) -> pd.DataFrame:
        """
        Aggregate net cashflows per payment_date across all lines.
        Accepts either get_net_cashflows() -> Series or get_cashflows() -> Dict.
        """
        rows = []
        for ln in position.lines:
            inst = ln.instrument
            units = float(ln.units)
            s = getattr(inst, "get_net_cashflows", None)
            if callable(s):
                ser = s()
                if ser is not None and not getattr(ser, "empty", True):
                    df = ser.to_frame("amount").reset_index()
                    if "payment_date" not in df.columns:
                        df = df.rename(columns={"index": "payment_date"})
                    df["amount"] = df["amount"].astype(float) * units
                    rows.append(df[["payment_date", "amount"]])
                    continue
            g = getattr(inst, "get_cashflows", None)
            if callable(g):
                flows = g() or {}
                flat = []
                for items in flows.values():
                    for cf in (items or []):
                        pay = cf.get("payment_date") or cf.get("date") or cf.get("pay_date") or cf.get("fixing_date")
                        amt = cf.get("amount")
                        if pay is None or amt is None:
                            continue
                        flat.append({"payment_date": pd.to_datetime(pay).date(), "amount": float(amt) * units})
                if flat:
                    rows.append(pd.DataFrame(flat))

        if not rows:
            return pd.DataFrame(columns=["payment_date", "amount"])
        df_all = pd.concat(rows, ignore_index=True)
        df_all["payment_date"] = pd.to_datetime(df_all["payment_date"]).dt.date
        return df_all.groupby("payment_date", as_index=False)["amount"].sum()

    @classmethod
    def position_carry_to_cutoff(cls, position: Position, valuation_date: dt.date, cutoff: dt.date) -> float:
        cf = cls._agg_net_cashflows(position)
        if cf.empty:
            return 0.0
        mask = (cf["payment_date"] > valuation_date) & (cf["payment_date"] <= cutoff)
        return float(cf.loc[mask, "amount"].sum())

    @classmethod
    def portfolio_style_stats(
        cls,
        base_position: Position,
        bumped_position: Position,
        *,
        valuation_date: dt.date,
        cutoff: dt.date,
    ) -> Dict[str, float]:
        """
        Same output shape as your `portfolio_stats` helper.
        """
        npv_b = cls.position_total_npv(base_position)
        npv_u = cls.position_total_npv(bumped_position)
        carry_b = cls.position_carry_to_cutoff(base_position, valuation_date, cutoff)
        carry_u = cls.position_carry_to_cutoff(bumped_position, valuation_date, cutoff)
        return {
            "npv_base": npv_b,
            "npv_bumped": npv_u,
            "npv_delta": npv_u - npv_b,
            "carry_base": carry_b,
            "carry_bumped": carry_u,
            "carry_delta": carry_u - carry_b,
        }

    # -------------------------------------------------------------------------
    # Per‑line NPV table (pure DataFrame; callers can render however they like)
    # -------------------------------------------------------------------------
    @staticmethod
    def npv_table_dataframe(
        *,
        position: Position,
        instrument_hash_to_asset: Optional[Mapping[object, Any]] = None,  # map(content_hash_obj -> Asset)
        bumped_position: Optional[Position] = None,
    ) -> pd.DataFrame:
        """
        Build a per-line NPV table. No portfolio columns here.
        - unique_identifier is filled only if instrument_hash_to_asset is provided
          and keyed by the *content_hash object* (not string), matching your code.
        """
        # Optional bumped lookup by content hash (string)
        bump_idx: Dict[str, Any] = {}
        if bumped_position is not None:
            for bl in bumped_position.lines:
                try:
                    bump_idx[str(bl.instrument.content_hash())] = bl
                except Exception:
                    pass

        rows = []
        for ln in position.lines:
            inst = ln.instrument
            units = float(getattr(ln, "units", 0.0) or 0.0)

            # hash object + string
            try:
                ch_obj = inst.content_hash()
                ch_str = str(ch_obj)
            except Exception:
                ch_obj, ch_str = None, None

            # base
            try:
                price_base = float(inst.price())
            except Exception:
                price_base = float("nan")
            npv_base = units * price_base if price_base == price_base else float("nan")

            # bumped (if provided)
            price_bumped = None
            npv_bumped = None
            npv_delta = None
            if ch_str is not None and ch_str in bump_idx:
                try:
                    price_bumped = float(bump_idx[ch_str].instrument.price())
                    npv_bumped = units * price_bumped
                    npv_delta = npv_bumped - npv_base
                except Exception:
                    pass

            # optional UID via provided map
            uid = None
            if instrument_hash_to_asset is not None and ch_obj in instrument_hash_to_asset:
                try:
                    uid = getattr(instrument_hash_to_asset[ch_obj], "unique_identifier", None)
                except Exception:
                    uid = None

            ms_asset_id = getattr(inst, "main_sequence_asset_id", None)
            inst_type = getattr(inst, "instrument_type", type(inst).__name__)

            rows.append(
                {
                    "instrument_type": inst_type,
                    "unique_identifier": uid,
                    "content_hash": ch_str,
                    "ms_asset_id": ms_asset_id,
                    "units": units,
                    "price_base": price_base,
                    "price_bumped": price_bumped,
                    "npv_base": npv_base,
                    "npv_bumped": npv_bumped,
                    "npv_delta": npv_delta,
                }
            )

        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Lookups by content hash
    # -------------------------------------------------------------------------
    @staticmethod
    def index_lines_by_content_hash(position: Position) -> Dict[str, Any]:
        """Map content_hash (string) -> line."""
        out = {}
        for ln in position.lines:
            try:
                h = str(ln.instrument.content_hash())
            except Exception:
                continue
            out[h] = ln
        return out

    @staticmethod
    def find_line_by_content_hash(position: Position, content_hash_str: str):
        """Return the line for a given content_hash string, or None."""
        for ln in position.lines:
            try:
                if str(ln.instrument.content_hash()) == str(content_hash_str):
                    return ln
            except Exception:
                pass
        return None

    @classmethod
    def from_template(cls, template: Position, *, base_curves_by_index,
                      valuation_date: dt.date) -> "PositionOperations":
        op = cls(position_template=template)
        op.set_curves(base_curves_by_index=base_curves_by_index)
        op.set_valuation_date(valuation_date)
        return op

    def yield_overlay_points(
            self,
            *,
            position: Optional[Position] = None,
            valuation_date: Optional[dt.date] = None,
            label_fn: Optional[Callable[[Any], str]] = None,
    ) -> List[Dict[str, float]]:
        """
        Build overlay points from a Position for plotting on the par-curve chart.
        - Reads each line's `extra_market_info["yield"]` (decimal, e.g., 0.085).
        - Uses instrument.maturity_date to compute x (years) from the given valuation date.
        - Returns list of dicts: {"x": years_to_maturity, "y": ytm_percent, "label": str}.
        """
        pos = position or self.base_position
        if pos is None:
            raise RuntimeError("yield_overlay_points: provide `position` or instantiate base_position first.")

        d0 = valuation_date or self.valuation_date
        if d0 is None:
            raise RuntimeError("yield_overlay_points: provide `valuation_date` or set self.valuation_date first.")

        pts: List[Dict[str, float]] = []
        for line in pos.lines:
            info = getattr(line, "extra_market_info", None) or {}
            ytm = info.get("yield", None)
            if ytm is None:
                continue  # nothing to plot for this line

            ins = line.instrument
            mat = getattr(ins, "maturity_date", None)
            if mat is None:
                continue

            try:
                x_years = (mat - d0).days / 365.0
            except Exception:
                # maturity_date not a date? be conservative and skip
                continue

            # Default label: short content-hash; allow caller override
            if callable(label_fn):
                label = str(label_fn(line))
            else:
                try:
                    label = str(ins.content_hash())[:3]
                except Exception:
                    label = ""

            pts.append({"x": float(x_years), "y": float(ytm) * 100.0, "label": label})

        pts.sort(key=lambda p: p["x"])
        return pts
