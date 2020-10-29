"""Qcodes drivers for National Instrument arbitrary waveform generators.

Requires nifgen package
"""

import nifgen

import logging

from typing import Optional, Dict, List, cast

from qcodes import Instrument, InstrumentChannel, ChannelList

logger = logging.getLogger(__name__)


class NI_FGEN(Instrument):
    r"""
    This is the QCoDeS driver for National Instruments FGEN devices based
    on the NI-FGEN driver, using the ``nifgen`` module. ``Parameter`` s for
    specific instruments should be defined in subclasses.

    This main class mostly just maintains a reference to a
    ``nifgen.Session``.

    Tested with

    - NI PXI-5421

    Args:
        name: Qcodes name for this instrument
        resource: Network address or VISA alias for the instrument.
        name_mapping: Optional mapping from default ("raw") channel names to
            custom aliases
        reset_device: whether to reset the device on initialization
        niswitch_kw: other keyword arguments passed to the ``niswitch.Session``
            constructor
    """

    def __init__(self, name: str, resource: str,
                 channel_name = None,
                 reset_device: bool = False,
                 nifgen_options: Optional[Dict] = None,
                 **kwargs):

        super().__init__(name=name, **kwargs)

        self.session = Session(resource, channel_name=channel_name,
                               reset_device=reset_device,
                               options=**nifgen_options)

        self.snapshot(update=True)  # make all channels read their conenctions

        self.connect_message()
