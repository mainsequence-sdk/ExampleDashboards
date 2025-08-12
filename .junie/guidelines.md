# QuantLibDev Development Guidelines

## Build/Configuration Instructions

### Environment Setup
This project uses **pipenv** for dependency management. The project requires Python 3.10+ (note: the Pipfile incorrectly specifies Python 3.1, which should be updated to a modern version).

#### Initial Setup
```bash
# Install pipenv if not already installed
pip install pipenv

# Install project dependencies
pipenv install

# Activate the virtual environment
pipenv shell
```

#### Key Dependencies
- **QuantLib-Python**: Core quantitative finance library for pricing and risk calculations
- **NumPy**: Numerical computing support

### Project Structure
```
src/
├── instruments/           # Financial instrument definitions
│   ├── base_instrument.py    # Abstract base class
│   ├── european_option.py    # European option implementation
│   └── interest_rate_swap.py # Interest rate swap implementation
├── pricing_models/        # Pricing engines and models
│   ├── black_scholes.py      # Black-Scholes model
│   └── swap_pricer.py        # Swap pricing utilities
├── data_interface.py      # Mock data API (replace with real data source)
└── utils.py              # Date conversion utilities
```

### Running the Application
The main entry point supports command-line pricing of derivatives:

```bash
# European Option Example
python main.py --instrument european_option --underlying "AAPL" --strike 150 --maturity "2026-12-31" --option_type call

# Interest Rate Swap Example
python main.py --instrument interest_rate_swap --notional 10000000 --start 2025-07-23 --maturity 2030-07-23 --fixed_rate 0.055 --float_spread 0.001

# Swap with Cashflow Analysis
python main.py --instrument interest_rate_swap --notional 10000000 --start 2025-07-23 --maturity 2030-07-23 --fixed_rate 0.055 --valuation_date 2026-01-15 --analyze_cashflows
```

## Testing Information

### Test Framework
The project uses Python's built-in `unittest` framework for testing.

### Running Tests
```bash
# Run all tests
python -m unittest discover -s . -p "test_*.py"

# Run a specific test file
python test_example.py

# Run with verbose output
python -m unittest -v test_example.py
```

### Test Structure
Tests are organized by instrument type:
- `TestEuropeanOption`: Tests for option pricing and Greeks calculation
- `TestInterestRateSwap`: Tests for swap pricing and cashflow analysis

### Adding New Tests
1. Create test files following the pattern `test_*.py`
2. Import required modules and instrument classes
3. Use `unittest.TestCase` as the base class
4. For swap tests, include `setUp()` method to clear QuantLib index histories:
   ```python
   def setUp(self):
       """Clear any existing fixings before each test."""
       ql.IndexManager.instance().clearHistories()
   ```

### Test Example
```python
import unittest
import datetime
from src.instruments.european_option import EuropeanOption

class TestNewInstrument(unittest.TestCase):
    def test_pricing(self):
        instrument = EuropeanOption(
            underlying="TEST",
            strike=100.0,
            maturity=datetime.date(2026, 12, 31),
            option_type="call"
        )
        price = instrument.price()
        self.assertIsInstance(price, float)
        self.assertGreater(price, 0)
```

## Additional Development Information

### Code Architecture

#### Design Patterns
- **Strategy Pattern**: Different pricing models (Black-Scholes, swap pricer) implement specific pricing strategies
- **Template Method**: Base instrument class defines common interface, concrete classes implement specific behavior
- **Facade Pattern**: Main.py provides simplified interface to complex pricing subsystems

#### Key Abstractions
- `DerivativeInstrument`: Base class for all financial instruments
- `APITimeSeries`: Data interface abstraction (currently mock, should be replaced with real data source)
- Pricing models are separated from instrument definitions for modularity

### QuantLib Integration Notes

#### Date Handling
- Use `src.utils.to_ql_date()` and `to_py_date()` for consistent date conversion
- Always set `ql.Settings.instance().evaluationDate` before pricing
- QuantLib uses its own Date class which differs from Python datetime

#### Index Management
- **Critical**: Clear index histories between tests using `ql.IndexManager.instance().clearHistories()`
- QuantLib doesn't allow duplicate fixings for the same date
- Historical fixings are automatically fetched and added during swap pricing

#### Yield Curve Construction
- The project uses piecewise log-cubic discount curve bootstrapping
- Market data includes both deposit rates (short-term) and swap rates (long-term)
- Curves are built from mock data but follow realistic market conventions

### Data Interface
The current `APITimeSeries` class provides mock data. In production:
- Replace with real market data provider (Bloomberg, Refinitiv, etc.)
- Implement proper error handling and data validation
- Add caching mechanisms for performance
- Consider async data fetching for multiple instruments

### Performance Considerations
- QuantLib objects are computationally expensive to create
- Cache yield curves and market data when possible
- Use QuantLib's built-in optimization features (e.g., `enableExtrapolation()`)
- Consider parallel processing for portfolio-level calculations

### Common Pitfalls
1. **Date Mismatches**: Always ensure consistent date handling between Python and QuantLib
2. **Index Fixings**: Clear histories between tests to avoid duplicate fixing errors
3. **Calendar Alignment**: Ensure business day calendars match the instrument's market
4. **Memory Management**: QuantLib objects can consume significant memory; clean up when possible

### Debugging Tips
- Enable QuantLib's detailed error messages for troubleshooting
- Use the mock data API's debug output to trace data flow
- Verify yield curve construction by checking discount factors at key tenors
- For swap pricing issues, examine individual cashflows using `get_cashflows()`

### Future Enhancements
- Add support for exotic derivatives (barriers, Asian options, etc.)
- Implement Monte Carlo pricing engines
- Add portfolio-level risk calculations (VaR, expected shortfall)
- Create web API interface for remote pricing
- Add real-time market data integration
- Implement proper logging and monitoring