from typing import Protocol, runtime_checkable

@runtime_checkable
class Instrument(Protocol):
    """
    Any object with a .price() -> float is considered a derivative instrument.
    Using a Protocol avoids metaclass conflicts with Pydantic BaseModel.
    """
    def price(self) -> float: ...

