import argparse
import datetime
import QuantLib as ql
from src.instruments.european_option import EuropeanOption
from src.instruments.interest_rate_swap import InterestRateSwap
from src.instruments.fixed_rate_bond import FixedRateBond


def main():
    """
    Main function to run the derivative pricing application.
    Parses command-line arguments to determine which instrument to price.
    """
    parser = argparse.ArgumentParser(description="QuantLib Derivatives Pricing Engine",
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--instrument", type=str, required=True,
                        help="Type of instrument to price (e.g., 'european_option', 'interest_rate_swap')")

    # --- Option Arguments ---
    option_group = parser.add_argument_group('Option Arguments')
    option_group.add_argument("--underlying", type=str, help="Ticker of the underlying asset (e.g., 'AAPL')")
    option_group.add_argument("--strike", type=float, help="Strike price of the option")
    option_group.add_argument("--option_type", type=str, choices=['call', 'put'],
                              help="Type of the option ('call' or 'put')")

    # --- Swap Arguments ---
    swap_group = parser.add_argument_group('Swap Arguments')
    swap_group.add_argument("--notional", type=float, help="Notional amount for the swap")
    swap_group.add_argument("--fixed_rate", type=float, help="Fixed rate for the swap's fixed leg")
    swap_group.add_argument("--float_spread", type=float, help="Spread over the floating index (e.g., 0.001 for 10bps)")
    swap_group.add_argument("--analyze_cashflows", action='store_true',
                            help="If set, prints a cashflow analysis for the swap.")

    # --- Bond Arguments ---
    bond_group = parser.add_argument_group('Bond Arguments')
    bond_group.add_argument("--face_value", type=float, help="Face amount of the bond")
    bond_group.add_argument("--coupon", type=float, help="Annual coupon as decimal (e.g., 0.045)")
    bond_group.add_argument("--frequency", type=str, default="6M", help="Coupon frequency (default: 6M)")
    bond_group.add_argument("--daycount", type=str, default="30/360", help="Day count (default: 30/360 USA)")

    # --- Common Arguments for all instruments ---
    common_group = parser.add_argument_group('Common Arguments')
    common_group.add_argument("--start", type=str, help="Start/effective date in YYYY-MM-DD format")
    common_group.add_argument("--maturity", type=str, help="Maturity/termination date in YYYY-MM-DD format")
    common_group.add_argument("--valuation_date", type=str,
                              help="The date for which to value the instrument (YYYY-MM-DD). Defaults to today.")

    args = parser.parse_args()

    try:
        valuation_date = datetime.datetime.strptime(args.valuation_date,
                                                    "%Y-%m-%d").date() if args.valuation_date else datetime.date.today()

        if args.instrument.lower() == 'european_option':
            if not all([args.underlying, args.strike, args.maturity, args.option_type]):
                raise ValueError(
                    "For a European option, --underlying, --strike, --maturity, and --option_type are required.")

            maturity_date = datetime.datetime.strptime(args.maturity, "%Y-%m-%d").date()
            option = EuropeanOption(
                underlying=args.underlying, strike=args.strike,
                maturity=maturity_date, option_type=args.option_type
            )
            price = option.price()
            greeks = option.get_greeks()

            print(f"\n==========================================")
            print(f"  RESULTS FOR {args.underlying.upper()} {args.option_type.upper()} OPTION")
            print(f"==========================================")
            print(f"{'Metric':<10} | {'Value':>15}")
            print(f"---------------------------")
            print(f"{'Price':<10} | {price:>15.4f}")
            print(f"{'Delta':<10} | {greeks['delta']:>15.4f}")
            print(f"{'Gamma':<10} | {greeks['gamma']:>15.4f}")
            print(f"{'Vega':<10} | {greeks['vega']:>15.4f}")
            print(f"{'Theta':<10} | {greeks['theta']:>15.4f}")
            print(f"{'Rho':<10} | {greeks['rho']:>15.4f}")
            print(f"==========================================")
        elif args.instrument.lower() == 'fixed_rate_bond':
            if not all([args.face_value, args.coupon, args.start, args.maturity]):
                raise ValueError("For a bond, --face_value, --coupon, --start, and --maturity are required.")
            issue_date = datetime.datetime.strptime(args.start, "%Y-%m-%d").date()
            maturity_date = datetime.datetime.strptime(args.maturity, "%Y-%m-%d").date()

            # map daycount
            dc = ql.Thirty360(ql.Thirty360.USA) if (args.daycount or '').upper().startswith(
                "30/360") else ql.Actual365Fixed()

            bond = FixedRateBond(
                face_value=args.face_value,
                coupon_rate=args.coupon,
                issue_date=issue_date,
                maturity_date=maturity_date,
                coupon_frequency=ql.Period(args.frequency or "6M"),
                day_count=dc,
                valuation_date=valuation_date
            )

            price = bond.price()
            analytics = bond.analytics()

            print(f"\n==========================================")
            print(f"  RESULTS FOR FIXED RATE BOND")
            print(f"==========================================")
            print(f"Price (NPV): {price:,.2f}")
            print(f"Clean Price (per 100): {analytics['clean_price']:.4f}")
            print(f"Dirty Price (per 100): {analytics['dirty_price']:.4f}")
            print(f"Accrued Amount: {analytics['accrued_amount']:.6f}")
            print(f"==========================================")

        elif args.instrument.lower() == 'interest_rate_swap':
            if not all([args.notional, args.start, args.maturity, args.fixed_rate]):
                raise ValueError("For a swap, --notional, --start, --maturity, and --fixed_rate are required.")

            start_date = datetime.datetime.strptime(args.start, "%Y-%m-%d").date()
            maturity_date = datetime.datetime.strptime(args.maturity, "%Y-%m-%d").date()

            # For demonstration, we hardcode some conventions and the floating index
            yield_curve_handle = ql.YieldTermStructureHandle(
                ql.FlatForward(ql.Date(21, 7, 2025), 0.05, ql.Actual365Fixed()))
            ibor_index = ql.USDLibor(ql.Period('3M'), yield_curve_handle)

            swap = InterestRateSwap(
                notional=args.notional,
                start_date=start_date,
                maturity_date=maturity_date,
                fixed_rate=args.fixed_rate,
                fixed_leg_tenor=ql.Period('6M'),
                fixed_leg_convention=ql.Unadjusted,
                fixed_leg_daycount=ql.Thirty360(ql.Thirty360.USA),
                float_leg_tenor=ql.Period('3M'),
                float_leg_spread=args.float_spread or 0.0,
                float_leg_ibor_index=ibor_index,
                valuation_date=valuation_date
            )
            price = swap.price()

            print(f"\n==========================================")
            print(f"  RESULTS FOR INTEREST RATE SWAP")
            print(f"==========================================")
            print(f"Price (NPV): {price:,.2f}")
            print(f"==========================================")

            if args.analyze_cashflows:
                cashflows = swap.get_cashflows()
                print("\n--- Fixed Leg Cashflows ---")
                print(f"{'Payment Date':<15} | {'Amount':>15}")
                print("-" * 33)
                for cf in cashflows['fixed']:
                    print(f"{str(cf['payment_date']):<15} | {cf['amount']:>15,.2f}")

                print("\n--- Floating Leg Cashflows ---")
                print(f"{'Payment Date':<15} | {'Fixing Date':<15} | {'Rate':>10} | {'Amount':>15}")
                print("-" * 65)
                for cf in cashflows['floating']:
                    print(
                        f"{str(cf['payment_date']):<15} | {str(cf['fixing_date']):<15} | {cf['rate']:>10.4%} | {cf['amount']:>15,.2f}")
                print("\n")

        else:
            print(f"Error: Instrument '{args.instrument}' is not supported.")

    except (ValueError, TypeError) as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    # Example for European Option:
    # python main.py --instrument european_option --underlying "SPY" --strike 500 --maturity "2026-12-31" --option_type call

    # Example for Interest Rate Swap:
    # python main.py --instrument interest_rate_swap --notional 10000000 --start 2025-07-23 --maturity 2030-07-23 --fixed_rate 0.055 --float_spread 0.001

    # Example for Swap with cashflow analysis and a forward valuation date:
    # python main.py --instrument interest_rate_swap --notional 10000000 --start 2025-07-23 --maturity 2030-07-23 --fixed_rate 0.055 --valuation_date 2026-01-15 --analyze_cashflows
    main()