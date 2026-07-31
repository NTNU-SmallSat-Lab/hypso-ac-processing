#!/usr/bin/env python3

import os
import sys
import numpy as np
from pathlib import Path
import pandas as pd
import glob
import gc
from rich import print
from rich.panel import Panel
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/camerop/AC/hypso-package/hypso')
sys.path.insert(0, '/home/camerop/AC/hypso-package/hypso1_calibration')
sys.path.insert(0, '/home/camerop/AC/hypso-package/hypso2_calibration')

#sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso')
#sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso1_calibration')
#sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso2_calibration')

import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

from hypso import Hypso
from hypso.write import write_l1b_nc_file, write_l1c_nc_file, write_l1d_nc_file, write_l2a_nc_file, write_products_nc_file
from hypso.classification import decode_jon_cnn_labels, decode_jon_cnn_cloud_mask, decode_jon_cnn_water_mask, decode_jon_cnn_land_mask

#from hypso.aeronet_oc import build_aeronet_queries, format_capture_date

from hypso.aeronet_oc import aeronet_oc_detect_matchups, \
                            aeronet_oc_generate_matchup, \
                            aeronet_oc_matchup_load_hypso_data, \
                            process_aeronet, \
                            build_aeronet_queries, \
                            format_capture_date, \
                            process_hypso, \
                            match_hypso_data, \
                            get_column_prods, \
                            process_satellite, \
                            match_data, \
                            match_all_data

from hypso.write import write_aeronet_oc_matchup_nc_file

import earthaccess

#TEST_DIR = "/home/camerop/HYPSO_DATA_AERONET_TEST/aeronetvenice_2025-03-04T10-38-05Z/"
#TEST_DIR = "/home/camerop/HYPSO_DATA_AERONET_TEST/zeebrugge_2025-09-01T11-27-47Z"
#TEST_DIR = "/home/camerop/HYPSO_DATA_AOC/annapolis_2026-03-10T16-03-45Z"
TEST_DIR = "/home/camerop/HYPSO_DATA_AOC_TEST/annapolis_2026-03-10T16-03-45Z"
TEST_DIR = "/home/camerop/HYPSO_DATA_AOC/aeronetvenice_2025-06-22T10-46-15Z"
TEST_DIR = "/home/camerop/HYPSO_DATA_AOC/frohavet_2025-02-25T11-26-39Z"
HYPSO_DATA_DIR = "/home/camerop/HYPSO_DATA_SNR"
#HYPSO_DATA_DIR = "/home/camerop/HYPSO_DATA_AOC"
#HYPSO_DATA_DIR = "/home/camerop/HYPSO_DATA_OCSMART"
#HYPSO_DATA_DIR = "/home/camerop/HYPSO_DATA_AERONET_TEST"

OUTPUT_BASE_DIR = Path("/home/camerop/Output/")

GENERATE_FIGURES = False
WRITE_DATACUBE = False

#RAD_CAL_COEFFS_OPTIONS = ["moved", "original", "adjusted"]
RAD_CAL_COEFFS_OPTIONS = ["moved"]



TOGGLE_PROCESSING = True

APPLY_MASKS = False
if APPLY_MASKS:
    LABEL = "moved"
else:
    LABEL = "moved_unmasked"

AERONET_OC_PRECHECK = False

TOGGLE_OCSMART = False
TOGGLE_ACOLITE = True
TOGGLE_6SV1 = False
TOGGLE_SREM = False
TOGGLE_POLYMER = False
TOGGLE_DARK_PIXEL_SUBTRACTION = False

TOGGLE_RUN_AC = True
TOGGLE_READ_AC = True


POLYMER_INPUT_PRODUCT_LEVEL = "l1c" 
POLYMER_BASE_PATH = '/home/camerop/AC/Polymer_HYPSO_SRF_Oct_2025/'
POLYMER_PATH = '/home/camerop/AC/Polymer_HYPSO_SRF_Oct_2025/polymer-master-v5'
EOREAD_PATH = '/home/camerop/AC/Polymer_HYPSO_SRF_Oct_2025/eoread'
EOTOOLS_PATH = '/home/camerop/AC/Polymer_HYPSO_SRF_Oct_2025/eotools'
CORE_PATH = '/home/camerop/AC/Polymer_HYPSO_SRF_Oct_2025/core'

OCSMART_PATH = "/home/_shared/ARIEL/atmospheric_correction/OC-SMART/OC-SMART_with_HYPSO_9-29-25_release/"
ACOLITE_PATH = "/home/camerop/AC/ACOLITE/acolite"

DEM_PATH = ""

EARTHDATA_u = "cpenne"
EARTHDATA_p = "Dec1!onJG0@1LogoMen5un!"



S3_GRANULE = Path("/home/camerop/AC/S3A_OL_1_EFR____20250522T094106_20250522T094406_20250523T103850_0179_126_136_1800_PS1_O_NT_004.SEN3")


OUTPUT_DIR = Path("/home/camerop/Output")

def main():

    logging.info(Panel(f"Running Sentinel-3 processing for {Path(S3_GRANULE).name}.", title="S3 + HYPSO Processing", expand=False))
    logging.info(f"Processing started at {datetime.now()}")

    try:
        auth = earthaccess.login(persist=True)
        earthaccess_login = True
        logging.info("NASA Earthaccess login successful!")
    except earthaccess.LoginAttemptFailure:
        logging.warning("NASA Earthaccess login failed!")
        earthaccess_login = False 



    logging.info("Running POLYMER atmospheric correction")

    logging.info("POLYMER configuration:")
    logging.info(f"POLYMER_PATH: {POLYMER_PATH}")
    logging.info(f"EOREAD_PATH: {EOREAD_PATH}")
    logging.info(f"EOTOOLS_PATH {EOTOOLS_PATH}")
    logging.info(f"CORE_PATH: {CORE_PATH}")



    if POLYMER_PATH is not None:
        sys.path.insert(0, POLYMER_PATH)

    if EOTOOLS_PATH is not None:
        sys.path.insert(0, EOTOOLS_PATH)

    if EOREAD_PATH is not None:
        sys.path.insert(0, EOREAD_PATH)

    if CORE_PATH is not None:
        sys.path.insert(0, CORE_PATH)

    sys.path.insert(0, POLYMER_BASE_PATH)


    from eoread.hypso import Level1_HYPSO
    from polymer.level1_olci import Level1_OLCI
    from polymer.main_v5 import run_polymer, run_polymer_dataset, default_output_datasets

    #input_file = str(S3_GRANULE)
    input_file = S3_GRANULE

    print(input_file)

    optional_output_datasets = ["SPM"]

    if_exists = "overwrite"

    output_file = run_polymer(
        input_file,
        #Level1_OLCI(input_file),
        dir_out=str(OUTPUT_DIR),
        output_datasets=default_output_datasets + optional_output_datasets,
        if_exists = if_exists,
    )


    print(output_file)







    gc.collect()

    logging.info(f"Processing has completed sucessfully for S3 granule {S3_GRANULE}!")

            

    

   











if __name__ == "__main__":

    #if len(sys.argv) < 2 or len(sys.argv) > 2:
    #    print("Usage: python process_l1d_dir.py <nc_dir_path>")
        
    main()

    gc.collect()
    sys.exit(0)

