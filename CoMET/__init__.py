from .user_interface_layer import *
from .tracker_output_translation_layer import *

from CoMET.analysis.get_vars import get_var
from CoMET.analysis.create_analysis_object import *

from CoMET.MOAAP import moaap

from .wrf_load import *
from .wrf_tobac import *

from .mesonh_load import *
from .mesonh_tobac import *

from .nexrad_load import *
from .nexrad_tobac import *

from .goes_load import *
from .goes_tobac import *

print("=====Welcome To CoMET=====\n\n"+
"This project was supported in part by the U.S. Department of Energy, Office of Science, Office of Workforce Development for Teachers and Scientists (WDTS) under the Science Undergraduate Laboratory Internships Program (SULI) and by the Brookhaven National Laboratory (BNL), Environmental and Climate Sciences Department (ECSD) under the BNL Supplemental Undergraduate Research Program (SURP).\n\n"+
"==========================\n")