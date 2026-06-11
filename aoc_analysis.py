"""This script finds the best point for Hypso and PACE data to match the RadCalNet site.
It also saves the pace spectrum at the point to a file. To future computation."""


import os
import sys
import numpy as np
from pathlib import Path
import pandas as pd
import glob

sys.path.insert(0, '/home/camerop/AC/hypso-package/hypso')
sys.path.insert(0, '/home/camerop/AC/hypso-package/hypso1_calibration')
sys.path.insert(0, '/home/camerop/AC/hypso-package/hypso2_calibration')

from hypso import Hypso
from hypso.write import write_l1b_nc_file, write_l1c_nc_file, write_l1d_nc_file, write_l2a_nc_file, write_products_nc_file
from hypso.classification import decode_jon_cnn_labels, decode_jon_cnn_cloud_mask, decode_jon_cnn_water_mask, decode_jon_cnn_land_mask

from hypso.aeronet_oc import aeronet_oc_detect_matchups, \
                            aeronet_oc_generate_matchup, \
                            aeronet_oc_matchup_load_hypso_data, \
                            process_aeronet, \
                            build_aeronet_queries, \
                            process_hypso

