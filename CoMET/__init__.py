from .user_interface_layer import *
from .tracker_output_translation_layer import *

from .post_processor import *

from CoMET.analysis.get_vars import get_var

from CoMET.MOAAP import moaap

from .wrfcube import *
from .wrf_load import *
from .wrf_tobac import *
from .wrf_moaap import *
from .wrf_calculate_products import *

from .mesonhcube import *
from .mesonh_load import *
from .mesonh_tobac import *
from .mesonh_moaap import *
from .mesonh_calculate_products import *

from .nexrad_load import *
from .nexrad_tobac import *

from .multi_nexrad_load import *
from .multi_nexrad_tobac import *

from .standard_radar_load import *
from .standard_radar_tobac import *

from .goes_load import *
from .goes_tobac import *

print(
    "=====Welcome To CoMET=====\n\n"
    + "This project was supported by the U.S. Department of Energy (DOE) Early Career Research Program, Atmospheric System Research (ASR) program, and the Office of Workforce Development for Teachers and Scientists (WDTS) under the Science Undergraduate Laboratory Internships Program (SULI).\n\n"
    + "If you are using this software for a publication, please cite: ####\n\n"
    "==========================\n"
)
