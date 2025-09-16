from .data_interface import DateInfo, MockDataInterface
import os

from .data_interface import MockDataInterface
data_interface=MockDataInterface
if "MAINSEQUENCE_TOKEN" in os.environ:
    from .data_interface import MSInterface

    data_interface = MSInterface()

