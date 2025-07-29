#from pkg_resources import DistributionNotFound
#from pkg_resources import get_distribution

#try:
#    __version__ = get_distribution(__name__).version
#except DistributionNotFound:
#    pass

#__all__ = [
#    "wavelet_base",
#    "wavesim",
#]

#from wavesim.wavesim import WaveSim
#from wavesim.wavelet_base import Wavelet2DBaseTorch

from wavesim import WaveSim
from wavelet_base import Wavelet2DBaseTorch