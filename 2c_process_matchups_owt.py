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

sys.path.insert(0, '/home/camerop/AC/hypso-package/hypso')
sys.path.insert(0, '/home/camerop/AC/hypso-package/hypso1_calibration')
sys.path.insert(0, '/home/camerop/AC/hypso-package/hypso2_calibration')

#sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso')
#sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso1_calibration')
#sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso2_calibration')




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
                            match_all_data, \
                            process_hypso_convolved

from hypso.write import write_aeronet_oc_matchup_nc_file


#sys.path.insert(0, '/home/camerop/AC/pyOWT/pyowt')
#from pyowt.OWT import OWT
#from pyowt.OpticalVariables import OpticalVariables



import earthaccess

import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

#TEST_DIR = "/home/camerop/HYPSO_DATA_AERONET_TEST/aeronetvenice_2025-03-04T10-38-05Z/"
#TEST_DIR = "/home/camerop/HYPSO_DATA_AERONET_TEST/zeebrugge_2025-09-01T11-27-47Z"
#TEST_DIR = "/home/camerop/HYPSO_DATA_AOC/annapolis_2026-03-10T16-03-45Z"
TEST_DIR = "/home/camerop/HYPSO_DATA_AOC_TEST"
HYPSO_DATA_DIR = "/home/camerop/HYPSO_DATA_AOC_TEST"

#TEST_DIR = "/home/camerop/HYPSO_DATA_AOC"
HYPSO_DATA_DIR = "/home/camerop/HYPSO_DATA_AOC"

#HYPSO_DATA_DIR = "/home/camerop/HYPSO_DATA_AOC"
#HYPSO_DATA_DIR = "/home/camerop/HYPSO_DATA_OCSMART"
#HYPSO_DATA_DIR = "/home/camerop/HYPSO_DATA_AERONET_TEST"

OUTPUT_BASE_DIR = Path("/home/camerop/Output/")

PYOWT_PATH = "/home/camerop/AC/pyOWT/"

GENERATE_FIGURES = False
WRITE_DATACUBE = False

#RAD_CAL_COEFFS_OPTIONS = ["moved", "original", "adjusted"]
RAD_CAL_COEFFS_OPTIONS = ["moved"]


AERONET_OC_PRECHECK = True





AERONET_OC_DATA_DIR = "/home/camerop/AC/AERONET_OC_Data"
AERONET_OC_SITES_CSV_PATH = "/home/camerop/AC/hypso-ac-processing/config/AERONET_OC_Sites.csv"



EARTHDATA_u = "cpenne"
EARTHDATA_p = "Dec1!onJG0@1LogoMen5un!"



if True:
    ATMOSPHERIC_CORRECTION_ALGS=["dps", "l1d"]
    LABEL = "moved_unmasked"
else:
    ATMOSPHERIC_CORRECTION_ALGS=["polymer", "acolite_l2w"]
    LABEL = "moved"








def find_matching_files(aeronet_oc_data_dir, label=None, product_level="l2a", atmospheric_correction_algorithms=["polymer", "acolite_l2w", "dps"]):
    """
    Loop through directories in AERONET_OC_DATA_DIR, and find the matching
    netCDF file that starts with the directory name.
    
    Args:
        aeronet_oc_data_dir: Path to the AERONET_OC_DATA_DIR
    
    Returns:
        List of full paths to matching files
    """
    matching_captures_dict = {}
    
    logging.info(f"Running search for {product_level} captures processed using {atmospheric_correction_algorithms}")
    logging.info(f"Provided capture label is {label}")

    # Loop through all items in the main directory
    for item in os.listdir(aeronet_oc_data_dir):

        logging.info(f"Searching capture {item} for {product_level} product files...")

        matched_files_in_dir = {}

        for ac_alg in atmospheric_correction_algorithms:


            dir_path = os.path.join(aeronet_oc_data_dir, item)


            # Pattern for the file: dirname + "-moved-l2a-polymer.nc"
            # The dirname format example: "aeronetvenice_2025-05-14T10-45-06Z"
            if ac_alg == "l1d":
                pattern = os.path.join(dir_path, f"{item}-{label}-l1d.nc")
            else:
                pattern = os.path.join(dir_path, f"{item}-{label}-{product_level}-{ac_alg}.nc")

            
            
            # Check if it's a directory
            if os.path.isdir(dir_path):
                

                # Search for matching files
                matching_files_list = glob.glob(pattern)

            try:
                matched_files_in_dir[ac_alg] = matching_files_list[0]
                logging.info(f"> Matching file found for '{item}' generated using {ac_alg}")
            except:
                logging.info(f"> No matching file found for '{item}' generated using {ac_alg}")
                
            #try:
            #    matched_files_in_dir["l1d"] = matching_l1d_files_list[0]
            #    logging.info(f"> Associated L1d file found for '{item}' generated using {ac_alg}")
            #except:
            #    logging.info(f"> No associated L1d file found for '{item}' generated using {ac_alg}")


        if len(matched_files_in_dir) > 0:

            matching_captures_dict[item] = matched_files_in_dir    

    logging.info(f"{len(matching_captures_dict.keys())} captures with {product_level} products found!")

    return matching_captures_dict



def main(coeff_type=None):

    print(Panel(f"Running matchup processing", title="HYPSO Processing", expand=False))
    logging.info(f"Processing started at {datetime.now()}")

    try:
        auth = earthaccess.login(persist=True)
        earthaccess_login = True
        logging.info("NASA Earthaccess login successful!")
    except earthaccess.LoginAttemptFailure:
        logging.info("NASA Earthaccess login failed!")
        earthaccess_login = False 


    

    


    logging.info("Entering AERONET-OC matchups code")

    output_dir = Path(OUTPUT_BASE_DIR, "AERONET-OC/")
    output_dir.mkdir(parents=True, exist_ok=True)

    matchup_lists = {}

    for aca in ATMOSPHERIC_CORRECTION_ALGS:
        matchup_lists[aca] = []


    matching_captures_dict = find_matching_files(HYPSO_DATA_DIR, label=LABEL, product_level="l2a", atmospheric_correction_algorithms=ATMOSPHERIC_CORRECTION_ALGS)


    #matching_files = ["/home/camerop/HYPSO_DATA_AOC/annapolis_2026-04-11T15-46-47Z/annapolis_2026-04-11T15-46-47Z-moved-l2a-polymer.nc",
    #"/home/camerop/HYPSO_DATA_AOC/aeronetvenice_2025-05-14T10-45-06Z/aeronetvenice_2025-05-14T10-45-06Z-moved-l2a-polymer.nc"]

    #matching_files = ["/home/camerop/HYPSO_DATA_AOC/annapolis_2025-05-17T15-51-35Z/annapolis_2025-05-17T15-51-35Z-moved-l2a-polymer.nc"]

    hypso_pace_matchups_df = None

    for matching_capture in matching_captures_dict.keys():

        #print(matching_capture)

        try:
            matching_capture_files = matching_captures_dict[matching_capture]

            #ATMOSPHERIC_CORRECTION_ALGS = list(matching_capture_files.keys())

            primary_file = matching_capture_files[ATMOSPHERIC_CORRECTION_ALGS[0]]

            if not os.path.isfile(primary_file):
                logging.error(f"The file '{primary_file}' does not exist.")
                continue

            satobj = Hypso(path=primary_file, verbose=True)
            aoc_queries = build_aeronet_queries(satobj)
            #hypso_wavelengths = satobj.wavelengths
            search_date = satobj.capture_datetime.strftime('%Y-%m-%d')
            local_path = Path(satobj.capture_dir)

            del satobj

        except Exception as ex:
            logging.warning(f"Could not load primary file for capture {matching_capture}! Skipping.")
            logging.warning(ex)
            continue




            


        try:
        
            aoc_full_wavelengths = []
            for aoc_query in aoc_queries:

                #aoc_cb = process_aeronet(aoc_site="Casablanca_Platform", 
                #                start_date="2024-06-01", end_date="2024-07-31",
                #                data_level=15)

                aoc_cb, aoc_wavelengths = process_aeronet(**aoc_query, pyowt_path=PYOWT_PATH)   

                aoc_full_wavelengths = list(set(aoc_full_wavelengths) | set(aoc_wavelengths))

                # Pull out coordinates 
                aoc_lat = aoc_cb["aoc_latitude"][0]
                aoc_lon = aoc_cb["aoc_longitude"][0]


                # HYPSO Matchups

                hypso_cb_dict = {}
                #hypso_convolved_cb_dict = {}

                for aca in ATMOSPHERIC_CORRECTION_ALGS:

                    try:
                        capture_file = matching_capture_files[aca]
                    except Exception as ex:
                        logging.info(f"No capture file to load for {aca} for {matching_capture}. Skipping.")
                        continue

                    logging.info(f"Loading data from {aca} {capture_file}")


                    if aca == "dps" or aca == "polymer":
                        divide_by_pi = True
                    else:
                        divide_by_pi = False

                    try:
                        satobj = Hypso(path=capture_file, verbose=True, label=LABEL)

                        # Regular HYPSO L2a
                        hypso_cb = process_hypso(satobj, aoc_lat, aoc_lon, atmospheric_correction=aca, divide_by_pi=divide_by_pi, pyowt_path=PYOWT_PATH)

                        # Convolve HYPSO with AERONET-OC SRFs (assumed to be Gaussian, 10nm FWHM)
                        hypso_convolved_cb = process_hypso_convolved(satobj, aoc_wavelengths, aoc_lat, aoc_lon, atmospheric_correction=aca, divide_by_pi=divide_by_pi, aeronet_fwhm=10.0, pyowt_path=PYOWT_PATH)


                        # Add L1D TOA reflectance
                        #l1d_nc_file = Path(satobj.capture_dir, satobj.l1d_nc_file)
                        #satobj = Hypso(path=l1d_nc_file, verbose=True)
                        #hypso_rhot_cb = process_hypso(satobj, aoc_lat, aoc_lon, atmospheric_correction="l1d", pyowt_path=PYOWT_PATH)
                        # Convolve HYPSO with AERONET-OC SRFs (assumed to be Gaussian, 10nm FWHM)
                        #hypso_rhot_convolved_cb = process_hypso_convolved(satobj, aoc_wavelengths, aoc_lat, aoc_lon, atmospheric_correction="l1d", aeronet_fwhm=10.0, pyowt_path=PYOWT_PATH)

                        hypso_cb_dict[aca] = hypso_cb
                        hypso_cb_dict[aca + "_convolved"] = hypso_convolved_cb

                        del satobj

                    except Exception as ex:
                        logging.warning(f"Could not load matchup data from {aca} {capture_file}")
                        logging.warning(ex)
                        continue

                # PACE Matchups


                # Pull out unique days
                unique_days = aoc_cb["aoc_datetime"].dt.date.unique()
                unique_days_str = [day.strftime('%Y-%m-%d') for day in unique_days]
                

                pace_cb = process_satellite(start_date=search_date, end_date=search_date,
                                latitude=aoc_lat, longitude=aoc_lon, sat="PACE",
                                selected_dates=unique_days_str,
                                local_path=local_path)

                for aca in ATMOSPHERIC_CORRECTION_ALGS:

                    try:
                        hypso_cb = hypso_cb_dict[aca]
                        hypso_convolved_cb = hypso_cb_dict[aca + "_convolved"]
                    except Exception as ex:
                        continue
                    

                    matched_data = match_all_data(aoc_cb, hypso_cb, df_pace=pace_cb, df_hypso_convolved=hypso_convolved_cb,
                            cv_max_hypso=0.4, cv_max_pace=0.15, senz_max=70.0,
                            min_percent_valid=50.0, max_time_diff=180, std_max=1.5, atmospheric_correction=aca)

                    #all_hypso_pace_dfs.append(matched_data)
                    matchup_lists[aca].append(matched_data)


                for aca in ATMOSPHERIC_CORRECTION_ALGS:

                    try:
                        matchup_list = matchup_lists[aca]
                    except Exception as ex:
                        continue

                    try:
                        
                        matchup_list_df = pd.concat(matchup_list, ignore_index=True)
                        logging.info("Updated matchup dataframe:")
                        print(matchup_list_df)
                    except ValueError:
                        print("No rows in dataframe!")
                        continue

                    if matchup_list_df is not None:
                        matchup_list_df.to_csv(Path(output_dir, f"aeronet_matchups_{aca}_tmp.csv"), index=False)
                        matchup_list_df.to_parquet(Path(output_dir, f"aeronet_matchups_{aca}_tmp.parquet"), index=False)

                #del satobj
            #del satobj

        except Exception as ex:
            logging.error("Exception occured")
            logging.error(ex)
            gc.collect()


    for aca in ATMOSPHERIC_CORRECTION_ALGS:

        try:
            matchup_list = matchup_lists[aca]
        except Exception as ex:
            continue

        try:
            matchup_list_df = pd.concat(matchup_list, ignore_index=True)
            logging.info(f"Final updated matchup dataframe")
            print(matchup_list_df)
        except ValueError:
            logging.warning("No rows in dataframe!")
            continue

        if matchup_list_df is not None:
            matchup_list_df.to_csv(Path(output_dir, f"aeronet_matchups_{aca}.csv"), index=False)
            matchup_list_df.to_parquet(Path(output_dir, f"aeronet_matchups_{aca}.parquet"), index=False)
    
    


    logging.info(f"Processing has completed sucessfully!")

    gc.collect()
    sys.exit(0)



   











if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        print("Usage: python 2c_process_matchups.py <nc_dir_path>")
        
        if TEST_DIR is not None:
            print("Attempting to use test dir")
            dir_path = TEST_DIR
        else:
            gc.collect()
            sys.exit(1)
    else:
        dir_path = sys.argv[1]


    for RAD_CAL_COEFFS in RAD_CAL_COEFFS_OPTIONS:


        main(coeff_type=RAD_CAL_COEFFS)

    gc.collect()
    sys.exit(0)

