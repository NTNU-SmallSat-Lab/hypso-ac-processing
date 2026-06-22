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
                            match_all_data

from hypso.write import write_aeronet_oc_matchup_nc_file

import earthaccess

#TEST_DIR = "/home/camerop/HYPSO_DATA_AERONET_TEST/aeronetvenice_2025-03-04T10-38-05Z/"
#TEST_DIR = "/home/camerop/HYPSO_DATA_AERONET_TEST/zeebrugge_2025-09-01T11-27-47Z"
#TEST_DIR = "/home/camerop/HYPSO_DATA_AOC/annapolis_2026-03-10T16-03-45Z"
TEST_DIR = "/home/camerop/HYPSO_DATA_AOC_TEST/annapolis_2026-03-10T16-03-45Z"
HYPSO_DATA_DIR = "/home/camerop/HYPSO_DATA_AOC_TEST"
#HYPSO_DATA_DIR = "/home/camerop/HYPSO_DATA_AOC"
#HYPSO_DATA_DIR = "/home/camerop/HYPSO_DATA_OCSMART"
#HYPSO_DATA_DIR = "/home/camerop/HYPSO_DATA_AERONET_TEST"

GENERATE_FIGURES = False
WRITE_DATACUBE = False

#RAD_CAL_COEFFS_OPTIONS = ["moved", "original", "adjusted"]
RAD_CAL_COEFFS_OPTIONS = ["moved"]

TOGGLE_PROCESSING = False
APPLY_MASKS = False

AERONET_OC_PRECHECK = True

TOGGLE_OCSMART = False
TOGGLE_ACOLITE = True
TOGGLE_6SV1 = False
TOGGLE_SREM = False
TOGGLE_POLYMER = False

TOGGLE_RUN_AC = True
TOGGLE_READ_AC = True

TOGGLE_AERONET_OC_MATCHUPS = False

MATCHUP_ATMOSPHERIC_CORRECTION = "polymer"
MATCHUP_ATMOSPHERIC_CORRECTION = "acolite"

POLYMER_INPUT_PRODUCT_LEVEL = "l1c" 
POLYMER_BASE_PATH = '/home/camerop/AC/Polymer_HYPSO_SRF_Oct_2025/'
POLYMER_PATH = '/home/camerop/AC/Polymer_HYPSO_SRF_Oct_2025/polymer-master-v5'
EOREAD_PATH = '/home/camerop/AC/Polymer_HYPSO_SRF_Oct_2025/eoread'
EOTOOLS_PATH = '/home/camerop/AC/Polymer_HYPSO_SRF_Oct_2025/eotools'
CORE_PATH = '/home/camerop/AC/Polymer_HYPSO_SRF_Oct_2025/core'

OCSMART_PATH = "/home/_shared/ARIEL/atmospheric_correction/OC-SMART/OC-SMART_with_HYPSO_9-29-25_release/"
ACOLITE_PATH = "/home/camerop/AC/ACOLITE/acolite"

DEM_PATH = ""

AERONET_OC_DATA_DIR = "/home/camerop/AC/AERONET_OC_Data"
AERONET_OC_SITES_CSV_PATH = "/home/camerop/AC/hypso-ac-processing/config/AERONET_OC_Sites.csv"



EARTHDATA_u = "cpenne"
EARTHDATA_p = "Dec1!onJG0@1LogoMen5un!"



def find_matching_files(aeronet_oc_data_dir, coeff_type="moved", product_level="l2a", atmospheric_correction="polymer"):
    """
    Loop through directories in AERONET_OC_DATA_DIR, and find the matching
    netCDF file that starts with the directory name.
    
    Args:
        aeronet_oc_data_dir: Path to the AERONET_OC_DATA_DIR
    
    Returns:
        List of full paths to matching files
    """
    matching_files = []
    
    # Loop through all items in the main directory
    for item in os.listdir(aeronet_oc_data_dir):
        dir_path = os.path.join(aeronet_oc_data_dir, item)
        
        # Check if it's a directory
        if os.path.isdir(dir_path):
            # Pattern for the file: dirname + "-moved-l2a-polymer.nc"
            # The dirname format example: "aeronetvenice_2025-05-14T10-45-06Z"
            pattern = os.path.join(dir_path, f"{item}*-{coeff_type}-{product_level}-{atmospheric_correction}.nc")
            
            # Search for matching files
            matching_files_list = glob.glob(pattern)
            
            if matching_files_list:
                # Add all matches to the list
                matching_files.extend(matching_files_list)
                for match in matching_files_list:
                    print(f"Found file: {match}")
            else:
                print(f"No matching file found for directory '{item}'")
    
    return matching_files



def main(l1a_nc_path, l1b_nc_path, lats_path=None, lons_path=None, coeff_type=None):

    print(Panel(f"Running processing for {Path(l1a_nc_path).name}.", title="HYPSO Processing", expand=False))
    print(f"Processing started at {datetime.now()}")

    try:
        auth = earthaccess.login(persist=True)
        earthaccess_login = True
        print("NASA Earthaccess login successful!")
    except earthaccess.LoginAttemptFailure:
        print("NASA Earthaccess login failed!")
        earthaccess_login = False 


    


    if TOGGLE_AERONET_OC_MATCHUPS:

        print("[INFO] Entering AERONET-OC matchups code")



        all_hypso_pace_dfs = []
        all_hypso_dfs = []
        all_pace_dfs = []

        matchups_df = None
        hypso_matchups_df = None
        pace_matchups_df = None


        matching_files = find_matching_files(HYPSO_DATA_DIR, coeff_type="moved", product_level="l2a", atmospheric_correction=MATCHUP_ATMOSPHERIC_CORRECTION)

        #matching_files = ["/home/camerop/HYPSO_DATA_AOC/annapolis_2026-04-11T15-46-47Z/annapolis_2026-04-11T15-46-47Z-moved-l2a-polymer.nc",
        #"/home/camerop/HYPSO_DATA_AOC/aeronetvenice_2025-05-14T10-45-06Z/aeronetvenice_2025-05-14T10-45-06Z-moved-l2a-polymer.nc"]

        #matching_files = ["/home/camerop/HYPSO_DATA_AOC/annapolis_2025-05-17T15-51-35Z/annapolis_2025-05-17T15-51-35Z-moved-l2a-polymer.nc"]

        for matching_file in matching_files:
        
            print(matching_file)

            if not os.path.isfile(matching_file):
                print(f"Error: The file '{matching_file}' does not exist.")
                continue


            try:
            
                satobj = Hypso(path=matching_file, verbose=True)

                aoc_queries = build_aeronet_queries(satobj)
                

                for aoc_query in aoc_queries:

                    #aoc_cb = process_aeronet(aoc_site="Casablanca_Platform", 
                    #                start_date="2024-06-01", end_date="2024-07-31",
                    #                data_level=15)

                    aoc_cb = process_aeronet(**aoc_query)
                    #print(aoc_cb.head())      


                    # Pull out coordinates 
                    aoc_lat = aoc_cb["aoc_latitude"][0]
                    aoc_lon = aoc_cb["aoc_longitude"][0]




                    # HYPSO Matchups

                    hypso_cb = process_hypso(satobj, aoc_lat, aoc_lon, atmospheric_correction=MATCHUP_ATMOSPHERIC_CORRECTION)


                    # PACE Matchups


                    # Pull out unique days
                    unique_days = aoc_cb["aoc_datetime"].dt.date.unique()
                    unique_days_str = [day.strftime('%Y-%m-%d') for day in unique_days]
                    search_date = satobj.capture_datetime.strftime('%Y-%m-%d')

                    pace_cb = process_satellite(start_date=search_date, end_date=search_date,
                                    latitude=aoc_lat, longitude=aoc_lon, sat="PACE",
                                    selected_dates=unique_days_str,
                                    local_path=Path(satobj.capture_dir))


                    hypso_pace_matchups = match_all_data(aoc_cb, hypso_cb, df_pace=pace_cb,
                            cv_max_hypso=0.4, cv_max_pace=0.15, senz_max=70.0,
                            min_percent_valid=50.0, max_time_diff=180, std_max=1.5)


                    #hypso_matchups = match_hypso_data(hypso_cb, aoc_cb, cv_max=0.60, senz_max=60.0, 
                    #    min_percent_valid=55.0, max_time_diff=180, std_max=3)

                    #pace_matchups = match_data(pace_cb, aoc_cb, cv_max=0.15, senz_max=60.0,
                    #    min_percent_valid=55.0, max_time_diff=180, std_max=1.5)

                    all_hypso_pace_dfs.append(hypso_pace_matchups)
                    #all_hypso_dfs.append(hypso_matchups)
                    #all_pace_dfs.append(pace_matchups)

                    #try:
                    #    hypso_pace_matchups_df = pd.concat(all_hypso_pace_dfs, ignore_index=True)
                    #    print(hypso_pace_matchups_df.head())
                    #except ValueError:
                    #    print("No rows in dataframe!")

                    '''
                    dict_aoc = get_column_prods(hypso_matchups_df, "aoc")
                    waves_aoc = np.array(dict_aoc["rrs"]["wavelengths"])
                    rrs_aoc = hypso_matchups_df[dict_aoc["rrs"]["columns"]].to_numpy()

                    dict_hypso = get_column_prods(hypso_matchups_df, "hypso")
                    waves_hypso = np.array(dict_hypso["rrs"]["wavelengths"])
                    rrs_hypso = hypso_matchups_df[dict_hypso["rrs"]["columns"]].to_numpy()
                    '''

                    try:
                        hypso_pace_matchups_df = pd.concat(all_hypso_pace_dfs, ignore_index=True)
                        print(hypso_pace_matchups_df)
                    except ValueError:
                        print("No rows in dataframe!")

                    if hypso_pace_matchups_df is not None:
                        hypso_pace_matchups_df.to_csv(f"aeronet_matchups_{MATCHUP_ATMOSPHERIC_CORRECTION}_tmp.csv", index=False)
                        hypso_pace_matchups_df.to_parquet(f"aeronet_matchups_{MATCHUP_ATMOSPHERIC_CORRECTION}_tmp.parquet", index=False)


            except Exception as ex:
                print("Exception occured")
                print(ex)


        try:
            hypso_pace_matchups_df = pd.concat(all_hypso_pace_dfs, ignore_index=True)
            print(hypso_pace_matchups_df)
        except ValueError:
            print("No rows in dataframe!")

        if hypso_pace_matchups_df is not None:
            hypso_pace_matchups_df.to_csv(f"aeronet_matchups_{MATCHUP_ATMOSPHERIC_CORRECTION}.csv", index=False)
            hypso_pace_matchups_df.to_parquet(f"aeronet_matchups_{MATCHUP_ATMOSPHERIC_CORRECTION}.parquet", index=False)
        
        
    print(f"Processing has completed sucessfully for capture {processing_capture_name}!")

            

    

   











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

        base_path = dir_path.rstrip('/')

        folder_name = os.path.basename(base_path)
        l1a_nc_path = os.path.join(base_path, f"{folder_name}-l1a.nc")
        l1b_nc_path = os.path.join(base_path, f"{folder_name}-{RAD_CAL_COEFFS}-l1b.nc")
        l1c_nc_path = os.path.join(base_path, f"{folder_name}-{RAD_CAL_COEFFS}-l1c.nc")
        l1d_nc_path = os.path.join(base_path, f"{folder_name}-{RAD_CAL_COEFFS}-l1d.nc")
        lats_path = os.path.join(base_path, "processing-temp", "latitudes_indirectgeoref.dat")
        lons_path = os.path.join(base_path, "processing-temp", "longitudes_indirectgeoref.dat")

        main(l1a_nc_path, l1b_nc_path, lats_path, lons_path, coeff_type=RAD_CAL_COEFFS)

    gc.collect()
    sys.exit(0)

