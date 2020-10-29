try:
    import seabreeze
    #seabreeze.use('pyseabreeze')
    import seabreeze.spectrometers as sb
    from seabreeze.spectrometers import Spectrometer
except ImportError:
    raise ImportError('Could not find seabreeze module.')

from qcodes.instrument.base import Instrument
from qcodes import ArrayParameter, MultiParameter, ManualParameter, ParameterWithSetpoints
from qcodes.utils import validators as vals

import numpy as np

class QEPro(Instrument):
    """
    QCoDeS driver for Ocean Optics QEPro spectrometer.

    Requires seabreeze module to be installed (pip install seabreeze).
    """
    def __init__(self, name: str, device_name: str = 'PXI-6251', **kwargs):
        super().__init__(name, **kwargs)

        self._spec =  Spectrometer.from_first_available()
        self._wl_nm_list = self._spec.wavelengths()
        self._int_time_micros_limits = self._spec.integration_time_micros_limits
        self._max_intensity = self._spec.max_intensity

        self.add_parameter(name='integration_time',
                           label='integration time',
                           unit='us',
                           set_cmd=lambda x: self._spec.integration_time_micros(x)
                           )

        self.add_parameter(name='correct_dark_counts',
                           label='correct dark counts boolean',
                           vals=vals.Bool(),
                           initial_value=False,
                           parameter_class=ManualParameter
                           )

        self.add_parameter(name='correct_nonlinearity',
                           label='correct nonlinearity boolean',
                           vals=vals.Bool(),
                           initial_value=False,
                           parameter_class=ManualParameter
                           )

        self.add_parameter(name='wavelengths',
                           label='wavelength',
                           unit='nm',
                           get_cmd=self._spec.wavelengths,
                           vals=vals.Arrays(shape=(len(self._wl_nm_list),))
                           )

        self.add_parameter(name='intensities',
                           label='Intensity',
                           unit='a.u.',
                           get_cmd=self._get_fresh_intensities,
                           vals=vals.Arrays(shape=(len(self._wl_nm_list),))
                           )

        self.add_parameter(name='spectrum',
                           label='Spectrum',
                           unit='a.u.',
                           parameter_class=ParameterWithSetpoints,
                           setpoints=(self.wavelengths,),
                           get_cmd=self._get_fresh_intensities,
                           vals=vals.Arrays(shape=(len(self._wl_nm_list),)) 
                           )

    def _get_fresh_intensities(self):
        self._spec.f.data_buffer.clear()
        return self._spec.intensities(correct_dark_counts=self.correct_dark_counts(), correct_nonlinearity=self.correct_nonlinearity())
