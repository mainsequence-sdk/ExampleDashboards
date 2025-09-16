# src/instruments/base_instrument.py
from typing import Protocol, runtime_checkable, Optional
from pydantic import BaseModel, Field
from .json_codec import JSONMixin

class InstrumentModel(BaseModel, JSONMixin):
    """
    Common base for all Pydantic instrument models.
    Adds a shared optional 'main_sequence_uid' field and shared config.
    """
    main_sequence_uid: Optional[str] = Field(
        default=None,
        description="Optional UID linking this instrument to a main sequence record."
    )

    # Keep your existing behavior (QuantLib types, etc.)
    model_config = {"arbitrary_types_allowed": True}
