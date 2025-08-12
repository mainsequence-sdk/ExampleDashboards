from abc import ABC, abstractmethod


class DerivativeInstrument(ABC):
    """
    Abstract base class for all derivative instruments.

    This class defines the common interface that all instruments must implement,
    ensuring that they can be priced in a consistent manner.
    """

    @abstractmethod
    def price(self) -> float:
        """
        Calculates the price (Net Present Value) of the instrument.

        This method must be implemented by all concrete instrument subclasses.
        It should encapsulate the entire pricing process, including data fetching,
        model setup, and calculation.
        """
        pass

