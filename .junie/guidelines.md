# Junie Development Guidelines: QuantLibDev

These guidelines will help you extend the `QuantLibDev` repository by adding new financial instruments and pricing models.

---

## 1. Project Overview & Architecture

The project is structured to separate financial **instruments**, their **pricing models**, and the **data interface**. A key principle is the use of **Pydantic** for instrument definitions to ensure data validation and clear schemas.

- **`src/instruments`**: This is where you define the financial products themselves. Each instrument is a **Pydantic `BaseModel`**, which provides data validation and a clear structure. These models also contain the methods for pricing.
- **`src/pricing_models`**: This contains the logic for valuing the instruments. For example, `black_scholes.py` is used for pricing European options, and `swap_pricer.py` is used for interest rate swaps.
- **`src/data_interface.py`**: This acts as a mock API to fetch the necessary market data for pricing, such as spot prices, volatility, or interest rates.
- **`main.py`**: This is the application's entry point, which parses command-line arguments to price a specified derivative.

---

## 2. How to Add a New Asset Instrument

Here’s the step-by-step process to define a new financial instrument, using a **Fixed Rate Bond** as an example.

### Step 1: Create the Instrument File

1.  Navigate to the `src/instruments/` directory.
2.  Create a new Python file for your instrument (e.g., `fixed_rate_bond.py`).

### Step 2: Define the Instrument Class using Pydantic

1.  Import necessary libraries: `datetime`, `ql` (QuantLib), and from `pydantic` import `BaseModel`, `Field`, and `PrivateAttr`.
2.  Import the corresponding pricing model you will create in the next section.
3.  Define a new class that inherits from **`pydantic.BaseModel`**.
4.  Define the instrument's attributes using standard type hints. Use **`Field(..., description="...")`** to provide helpful documentation for each attribute.
5.  Use **`PrivateAttr`** for any runtime objects (like the QuantLib instrument object) that should not be part of the model's schema.
6.  Add a `Config` inner class with `arbitrary_types_allowed = True` to permit QuantLib objects.

```python
# src/instruments/fixed_rate_bond.py
import datetime
import QuantLib as ql
from typing import Optional
from pydantic import BaseModel, Field, PrivateAttr
from src.pricing_models.bond_pricer import create_fixed_rate_bond

class FixedRateBond(BaseModel):
    """Fixed-rate bond (Pydantic model)."""

    face_value: float = Field(..., description="The principal amount of the bond.")
    coupon_rate: float = Field(..., description="The annual coupon rate as a decimal (e.g., 0.05 for 5%).")
    issue_date: datetime.date = Field(..., description="The date the bond was issued.")
    maturity_date: datetime.date = Field(..., description="The date the bond matures.")
    valuation_date: datetime.date = Field(
        default_factory=datetime.date.today,
        description="The date for which the bond is being valued."
    )

    # Private runtime attributes for QuantLib objects
    _bond: Optional[ql.FixedRateBond] = PrivateAttr(default=None)

    class Config:
        # Allow complex types like QuantLib objects
        arbitrary_types_allowed = True

    # ... Pricing methods will be defined here ...
```

### Step 3: Implement the Pricing and Analytics Methods

1.  **`_setup_pricer`**: Create a private helper method on the model to set up the QuantLib object. This method should call the pricer function you'll create (e.g., `create_fixed_rate_bond`).
2.  **`price()`**: Implement the pricing method. This should call `_setup_pricer()` and then use the private QuantLib object (e.g., `self._bond`) to calculate and return the Net Present Value (NPV).
3.  **`analytics()`** (Optional): Add other methods to return additional metrics, such as clean/dirty price or accrued interest.

---

## 3. How to Add a New Pricing Model

Here's how to create a new pricer for the **Fixed Rate Bond** instrument.

### Step 1: Create the Pricer File

1.  Navigate to the `src/pricing_models/` directory.
2.  Create a new Python file for your pricer (e.g., `bond_pricer.py`).

### Step 2: Implement the Data-Fetching and Curve-Building Logic

1.  Import the necessary libraries, especially `APITimeSeries` from the data interface.
2.  Create a function to build the necessary market data structures. For a bond, this would be a yield curve. This function should call `APITimeSeries.get_historical_data` to fetch the required rates.

```python
# src/pricing_models/bond_pricer.py
import QuantLib as ql
from src.data_interface import APITimeSeries

def build_bond_discount_curve(calculation_date: ql.Date) -> ql.YieldTermStructureHandle:
    """
    Builds a discount curve from mock zero rates.
    """
    print("Building discount curve from 'discount_bond_curve'...")
    data = APITimeSeries.get_historical_data("discount_bond_curve", {"USD_discount_curve": {}})
    # ... logic to build and return a QuantLib yield curve ...
```

### Step 3: Create the Main Pricing Function

1.  Create the main function that the instrument class will call (e.g., `create_fixed_rate_bond`).
2.  This function should:
    - Set the QuantLib evaluation date.
    - Call the curve-building function you created in the previous step.
    - Construct the QuantLib `FixedRateBond` object.
    - Create a pricing engine (e.g., `ql.DiscountingBondEngine`) and attach it to the bond object.
    - Return the fully configured QuantLib bond object.

---

## 4. Final Integration

To make your new instrument available through the command line:

1.  **Open `main.py`**.
2.  **Import** your new instrument class (e.g., `from src.instruments.fixed_rate_bond import FixedRateBond`).
3.  Add a new `elif` block in the `main` function to handle the new instrument.
4.  Inside this block:
    - Add command-line argument parsing for the new instrument's parameters.
    - Instantiate your new instrument model.
    - Call its `price()` and any other analytics methods.
    - Print the results in a formatted table.
