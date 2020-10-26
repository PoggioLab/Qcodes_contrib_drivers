try:
    import seabreeze.spectrometers as sb
    from seabreeze.spectrometers import Spectrometer
except ImportError:
    raise ImportError('Could not find seabreeze module.')

from qcodes.instrument.base import Instrument
from qcodes import ArrayParameter, MultiParameter, ManualParameter
from qcodes.utils import validators as vals

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

        self.add_parameter(name='wavelengths',
                           label='wavelength',
                           unit='nm',
                           get_cmd=lambda: self._spec.wavelengths()
                           )

        self.add_parameter(name='intensities',
                           label='intensity',
                           unit='a.u.',
                           get_cmd=lambda: self._spec.intensities()
                           )
