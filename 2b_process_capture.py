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







def main(l1a_nc_path, l1b_nc_path, lats_path=None, lons_path=None, coeff_type=None):

    logging.info(Panel(f"Running processing for {Path(l1a_nc_path).name}.", title="HYPSO Processing", expand=False))
    logging.info(f"Processing started at {datetime.now()}")

    try:
        auth = earthaccess.login(persist=True)
        earthaccess_login = True
        logging.info("NASA Earthaccess login successful!")
    except earthaccess.LoginAttemptFailure:
        logging.warning("NASA Earthaccess login failed!")
        earthaccess_login = False 


    if not APPLY_MASKS:
        processing_label = RAD_CAL_COEFFS + "_unmasked"
    else:
        processing_label = RAD_CAL_COEFFS

    if TOGGLE_PROCESSING:
        # Check if the first file exists
        if not os.path.isfile(l1a_nc_path):
            logging.error(f"The file '{l1a_nc_path}' does not exist.")
            return

        # Process the first file
        logging.info(f"Processing file: {l1a_nc_path}")

        nc_file = Path(l1a_nc_path)


        satobj = Hypso(path=nc_file, verbose=True, label=processing_label)

        if satobj.l1d_cube is None:

            # Run indirect georeferencing
            if lats_path is not None and lons_path is not None:
                try:

                    with open(lats_path, mode='rb') as file:
                        file_content = file.read()
                    
                    lats = np.frombuffer(file_content, dtype=np.float32)

                    lats = lats.reshape(satobj.spatial_dimensions)

                    with open(lons_path, mode='rb') as file:
                        file_content = file.read()
                    
                    lons = np.frombuffer(file_content, dtype=np.float32)
        
                    lons = lons.reshape(satobj.spatial_dimensions)


                    # Directly provide the indirect lat/lons loaded from the file. This function will run the track geometry computations.
                    satobj.run_georeferencing(latitudes=lats, longitudes=lons)

                    satobj.generate_l1b_cube(coeff_type=coeff_type)
                    
                    if False:
                        wls = np.around(np.array(satobj.spectral_coeffs),1)
                        wls = wls.astype(int)
                        print(wls)
                        sys.exit(1)

                    #satobj.generate_l1c_cube()
                    satobj.generate_l1d_cube(use_direct_georef=False, generate_figures=GENERATE_FIGURES)

                except Exception as ex:
                    logging.warning(ex)
                    logging.warning('Indirect georeferencing has failed. Defaulting to direct georeferencing.')

                    satobj.run_direct_georeferencing()
                    satobj.generate_l1b_cube(coeff_type=coeff_type)
                    #satobj.generate_l1c_cube()
                    satobj.generate_l1d_cube(use_direct_georef=True, generate_figures=GENERATE_FIGURES)

                    satobj.latitudes = satobj.latitudes_direct
                    satobj.longitudes = satobj.longitudes_direct

            else:
                satobj.run_direct_georeferencing()

                satobj.generate_l1b_cube(coeff_type=coeff_type)
                #satobj.generate_l1c_cube()
                satobj.generate_l1d_cube(use_direct_georef=True, generate_figures=GENERATE_FIGURES)


            if APPLY_MASKS:
                try:
                    # Load masks
                    spatial_dimensions = satobj.spatial_dimensions
                    full_path = os.path.join(Path(l1d_nc_path).parent, "processing-temp")
                    labels_path = os.path.join(full_path, "sea-land-cloud.labels")

                    labels_arr = decode_jon_cnn_labels(file_path=labels_path, spatial_dimensions=spatial_dimensions)
                    cloud_labels_arr = decode_jon_cnn_cloud_mask(file_path=labels_path, spatial_dimensions=spatial_dimensions)
                    water_labels_arr = decode_jon_cnn_water_mask(file_path=labels_path, spatial_dimensions=spatial_dimensions)
                    land_labels_arr = decode_jon_cnn_land_mask(file_path=labels_path, spatial_dimensions=spatial_dimensions)

                    satobj.land_mask = land_labels_arr
                    satobj.cloud_mask = cloud_labels_arr
                
                    masked=True
                    
                except Exception as ex:
                    logging.warning("Masking failed! No mask will be applied to the capture.")
                    logging.warning(ex)
                    masked=False
            else:
                masked=False


            satobj.compute_csiro_srfs()

            satobj.ac_polymer_generate_srf_nc()
            satobj.ac_polymer_generate_ssi_nc()
            satobj.ac_polymer_generate_esun_nc()

            datacube = WRITE_DATACUBE

            write_l1b_nc_file(satobj, overwrite=True, masked=masked, datacube=datacube) 
            write_l1c_nc_file(satobj, overwrite=True, masked=masked, datacube=datacube)
            write_l1d_nc_file(satobj, overwrite=True, masked=masked, datacube=datacube)

    else:

        # Check if the first file exists
        if not os.path.isfile(l1d_nc_path):
            logging.error(f"The file '{l1d_nc_path}' does not exist.")
            return

        # Process the first file
        logging.info(f"Processing file: {l1d_nc_path}")

        nc_file = Path(l1d_nc_path)

        satobj = Hypso(path=nc_file, verbose=True, label=processing_label)


    processing_capture_name = satobj.capture_name
 
    # Check for AERONET-OC data before proceeding to Atmospheric Correction
    if AERONET_OC_PRECHECK:

        satobj_precheck = Hypso(path=nc_file, verbose=True, load_cube = False)

        aoc_query = build_aeronet_queries(satobj_precheck)

        del satobj_precheck

        try:
            aoc_cb = process_aeronet(**aoc_query[0])
        except Exception as ex:
            logging.warning(ex)
            logging.warning(f"No AERONET-OC data are available for this capture for queried date and time.")
            logging.warning("Submitted AERONET-OC query:")
            print(aoc_query)
            logging.warning(f"The processing of capture {processing_capture_name} will now end.")
            gc.collect()
            sys.exit(0)



    # Atmospheric correction

    if TOGGLE_OCSMART:
        satobj.ocsmart_dir = OCSMART_PATH
        if TOGGLE_RUN_AC:

            logging.info("Running OC-SMART atmospheric correction")

            satobj.ac_ocsmart_stage_input()
            satobj.ac_ocsmart_run_correction()
        if TOGGLE_READ_AC:
            satobj.ac_ocsmart_open_output()
            write_l2a_nc_file(satobj=satobj, correction="ocsmart", overwrite=True, datacube=False)

    if TOGGLE_ACOLITE:
        satobj.acolite_dir = ACOLITE_PATH
        if TOGGLE_RUN_AC:

            logging.info("Running ACOLITE atmospheric correction")

            satobj.ac_acolite_run_correction(input_product_level='L1D', EARTHDATA_u=EARTHDATA_u, EARTHDATA_p=EARTHDATA_p)
        if TOGGLE_READ_AC:
            satobj.ac_acolite_open_output()
            write_l2a_nc_file(satobj=satobj, correction="acolite_l2r", overwrite=True, datacube=False)
            write_l2a_nc_file(satobj=satobj, correction="acolite_l2w", overwrite=True, datacube=False)

    if TOGGLE_6SV1:

        logging.info("Running 6SV1 atmospheric correction")

        from hypso.ac import run_6sv1_atmospheric_correction
        dem_path = Path("/home/cameron/Nedlastinger/GMTED2km.tif")

        cube = run_6sv1_atmospheric_correction(satobj, dem_path)

        satobj.l2_cube['6sv1'] = cube


        logging.info("Writing POLYMER atmospheric correction output file")
        logging.info("L2a file will be written to capture directory")
        write_l2a_nc_file(satobj, correction='6sv1', datacube=False, overwrite=True)

    if TOGGLE_POLYMER:

        if TOGGLE_RUN_AC:
            
            logging.info("Running POLYMER atmospheric correction")

            logging.info("POLYMER configuration:")
            logging.info(f"POLYMER_PATH: {POLYMER_PATH}")
            logging.info(f"EOREAD_PATH: {EOREAD_PATH}")
            logging.info(f"EOTOOLS_PATH {EOTOOLS_PATH}")
            logging.info(f"CORE_PATH: {CORE_PATH}")

            datasets = satobj.ac_polymer_run_correction(polymer_base_path=POLYMER_BASE_PATH,
                                                        polymer_path=POLYMER_PATH, 
                                                        eoread_path=EOREAD_PATH,
                                                        eotools_path=EOTOOLS_PATH,
                                                        core_path=CORE_PATH,
                                                        input_product_level=POLYMER_INPUT_PRODUCT_LEVEL)

            logging.info("POLYMER atmospheric correction finished")

        if TOGGLE_READ_AC:

            logging.info("Reading POLYMER atmospheric correction output file")
            logging.info("L2a and product files will be written to capture directory")

            datasets = satobj.ac_polymer_open_output(input_product_level=POLYMER_INPUT_PRODUCT_LEVEL)
            
            satobj.products['chla'] = datasets['chla']

            write_l2a_nc_file(satobj=satobj, correction="polymer", overwrite=True, datacube=False)
            write_products_nc_file(satobj, overwrite=True, file_name="polymer_chl.nc")

    if TOGGLE_DARK_PIXEL_SUBTRACTION:
        
        logging.info("Running Dark Pixel Subtraction (DPS) atmospheric correction")
        logging.info("L2a file will be written to capture directory")

        dark_pixel = satobj.ac_dark_pixel_subtraction()
        plt.plot(satobj.wavelengths, dark_pixel)
        plt.savefig(Path(satobj.capture_dir, "dark_pixel.png"))
        plt.close()
        write_l2a_nc_file(satobj, correction='dps', datacube=False, overwrite=True)


    gc.collect()

    logging.info(f"Processing has completed sucessfully for capture {processing_capture_name}!")

            

    

   











if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        print("Usage: python process_l1d_dir.py <nc_dir_path>")
        
        if TEST_DIR is not None:
            logging.info("Attempting to use test dir")
            dir_path = TEST_DIR
        else:
            gc.collect()
            sys.exit(1)
    else:
        dir_path = sys.argv[1]


    for RAD_CAL_COEFFS in RAD_CAL_COEFFS_OPTIONS:

        base_path = dir_path.rstrip('/')

        folder_name = os.path.basename(base_path)
        l1a_nc_path = os.path.join(base_path, f"{folder_name}-l1a.nc")
        l1b_nc_path = os.path.join(base_path, f"{folder_name}-{LABEL}-l1b.nc")
        l1c_nc_path = os.path.join(base_path, f"{folder_name}-{LABEL}-l1c.nc")
        l1d_nc_path = os.path.join(base_path, f"{folder_name}-{LABEL}-l1d.nc")
        lats_path = os.path.join(base_path, "processing-temp", "latitudes_indirectgeoref.dat")
        lons_path = os.path.join(base_path, "processing-temp", "longitudes_indirectgeoref.dat")

        main(l1a_nc_path, l1b_nc_path, lats_path, lons_path, coeff_type=RAD_CAL_COEFFS)

    gc.collect()
    sys.exit(0)

