#!/usr/bin/env python3

import os
import sys
import numpy as np
from pathlib import Path
import pandas as pd
import glob

sys.path.insert(0, '/home/camerop/AC/hypso-package/hypso')
sys.path.insert(0, '/home/camerop/AC/hypso-package/hypso1_calibration')
sys.path.insert(0, '/home/camerop/AC/hypso-package/hypso2_calibration')

#sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso')
#sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso1_calibration')
#sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso2_calibration')



from hypso import Hypso
from hypso.write import write_l1b_nc_file, write_l1c_nc_file, write_l1d_nc_file, write_l2a_nc_file, write_products_nc_file
from hypso.classification import decode_jon_cnn_labels, decode_jon_cnn_cloud_mask, decode_jon_cnn_water_mask, decode_jon_cnn_land_mask


TEST_DIR = "/home/camerop/HYPSO_DATA_AERONET/aeronetvenice_2025-05-14T10-45-06Z"

GENERATE_FIGURES = False
WRITE_DATACUBE = False

#RAD_CAL_COEFFS_OPTIONS = ["moved", "original", "adjusted"]
RAD_CAL_COEFFS_OPTIONS = ["moved"]

TOGGLE_PROCESSING = True
APPLY_MASKS = True

TOGGLE_OCSMART = False
TOGGLE_ACOLITE = False
TOGGLE_6SV1 = False
TOGGLE_SREM = False
TOGGLE_POLYMER = True

TOGGLE_RUN_AC = True
TOGGLE_READ_AC = True

TOGGLE_AERONET_OC_MATCHUPS = False
TOGGLE_AERONET_OC_MATCHUPS_2 = True

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

HYPSO_DATA_DIR = "/home/camerop/HYPSO_DATA_AOC"

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

    print(TOGGLE_PROCESSING)

    if TOGGLE_PROCESSING:
        # Check if the first file exists
        if not os.path.isfile(l1a_nc_path):
            print(f"Error: The file '{l1a_nc_path}' does not exist.")
            return

        # Process the first file
        print(f"Processing file: {l1a_nc_path}")

        nc_file = Path(l1a_nc_path)

        satobj = Hypso(path=nc_file, verbose=True, label=RAD_CAL_COEFFS)

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
                        exit()

                    #satobj.generate_l1c_cube()
                    satobj.generate_l1d_cube(use_direct_georef=False, generate_figures=GENERATE_FIGURES)

                except Exception as ex:
                    print(ex)
                    print('Indirect georeferencing has failed. Defaulting to direct georeferencing.')

                    satobj.run_direct_georeferencing()
                    satobj.generate_l1b_cube(coeff_type=coeff_type)
                    #satobj.generate_l1c_cube()
                    satobj.generate_l1d_cube(use_direct_georef=True, generate_figures=GENERATE_FIGURES)

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
                    print(ex)
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
            print(f"Error: The file '{l1d_nc_path}' does not exist.")
            return

        # Process the first file
        print(f"Processing file: {l1d_nc_path}")

        nc_file = Path(l1d_nc_path)

        satobj = Hypso(path=nc_file, verbose=True, label=RAD_CAL_COEFFS)

 


    # Atmospheric correction

    if TOGGLE_OCSMART:
        satobj.ocsmart_dir = OCSMART_PATH
        if TOGGLE_RUN_AC:
            satobj.ac_ocsmart_stage_input()
            satobj.ac_ocsmart_run_correction()
        if TOGGLE_READ_AC:
            satobj.ac_ocsmart_open_output()
            write_l2a_nc_file(satobj=satobj, correction="ocsmart", overwrite=True, datacube=False)

    if TOGGLE_ACOLITE:
        satobj.acolite_dir = ACOLITE_PATH
        if TOGGLE_RUN_AC:
            satobj.ac_acolite_run_correction(input_product_level='L1D', EARTHDATA_u=EARTHDATA_u, EARTHDATA_p=EARTHDATA_p)
        if TOGGLE_READ_AC:
            satobj.ac_acolite_open_output()
            write_l2a_nc_file(satobj=satobj, correction="acolite_l2r", overwrite=True, datacube=False)
            write_l2a_nc_file(satobj=satobj, correction="acolite_l2w", overwrite=True, datacube=False)

    if TOGGLE_6SV1:
        from hypso.ac import run_6sv1_atmospheric_correction
        dem_path = Path("/home/cameron/Nedlastinger/GMTED2km.tif")

        cube = run_6sv1_atmospheric_correction(satobj, dem_path)

        satobj.l2_cube['6sv1'] = cube

        write_l2a_nc_file(satobj, correction='6sv1', datacube=False, overwrite=True)

    if TOGGLE_POLYMER:

        if TOGGLE_RUN_AC:

            #print(POLYMER_PATH)
            #print(EOREAD_PATH)
            #print(EOTOOLS_PATH)
            #print(CORE_PATH)


            datasets = satobj.ac_polymer_run_correction(polymer_base_path=POLYMER_BASE_PATH,
                                                        polymer_path=POLYMER_PATH, 
                                                        eoread_path=EOREAD_PATH,
                                                        eotools_path=EOTOOLS_PATH,
                                                        core_path=CORE_PATH,
                                                        input_product_level=POLYMER_INPUT_PRODUCT_LEVEL)



        if TOGGLE_READ_AC:
            datasets = satobj.ac_polymer_open_output(input_product_level=POLYMER_INPUT_PRODUCT_LEVEL)
            
            satobj.products['chla'] = datasets['chla']

            write_l2a_nc_file(satobj=satobj, correction="polymer", overwrite=True, datacube=False)
            write_products_nc_file(satobj, overwrite=True, file_name="polymer_chl.nc")

    if TOGGLE_AERONET_OC_MATCHUPS:

        print("[INFO] Entering AERONET-OC matchups code")


        from hypso.aeronet_oc import aeronet_oc_detect_matchups, \
                                    aeronet_oc_generate_matchup, \
                                    aeronet_oc_matchup_load_hypso_data
        
        from hypso.write import write_aeronet_oc_matchup_nc_file

        satobj = Hypso(path=l1b_nc_path, load_cube = False, verbose=True)
        
        atmospheric_correction = "polymer"

        capture_dir = satobj.capture_dir

        l2a_name = satobj.l2a_name(label=RAD_CAL_COEFFS, 
                                   atmospheric_correction=atmospheric_correction)
        print(l2a_name)


        l2a_nc_path = Path(capture_dir, l2a_name)
        
        del satobj

        print(l2a_nc_path)

        if not os.path.isfile(l2a_nc_path):
            print(f"Error: The file '{l2a_nc_path}' does not exist.")
            return None

        # Process the first file
        print(f"Processing file for AERONET-OC: {l2a_nc_path}")

        try:
            satobj = Hypso(path=l2a_nc_path, verbose=True)


            matchups = aeronet_oc_detect_matchups(satobj, 
                                                AERONET_OC_SITES_CSV_PATH, 
                                                atmospheric_correction=atmospheric_correction)

            print(f"[INFO] Detected {len(matchups)} potential matchups.")
            print("[INFO] Combining matchups with HYPSO data.")
            for matchup_number, matchup in enumerate(matchups):

                matchup_aeronet_data = aeronet_oc_generate_matchup(satobj,
                                                    matchup,
                                                    AERONET_OC_DATA_DIR,
                                                    )

                if matchup_aeronet_data is None:
                    print(f"[INFO] Skipping matchup {matchup_number+1}")
                    continue

                print("[INFO] AERONET-OC matchup for L2a file")
                matchup_hypso_data = aeronet_oc_matchup_load_hypso_data(satobj, matchup, atmospheric_correction=atmospheric_correction, n_size=5)
                matchup_data = matchup_hypso_data | matchup_aeronet_data
                write_aeronet_oc_matchup_nc_file(satobj, matchup_data, atmospheric_correction=atmospheric_correction, datacube=True, matchup_number=matchup_number)

                print("[INFO] AERONET-OC matchup for L1d file")
                satobj = Hypso(path=l1d_nc_path, verbose=True)
                matchup_hypso_data = aeronet_oc_matchup_load_hypso_data(satobj, matchup, atmospheric_correction=atmospheric_correction, n_size=5)
                matchup_data = matchup_hypso_data | matchup_aeronet_data
                write_aeronet_oc_matchup_nc_file(satobj, matchup_data, atmospheric_correction=atmospheric_correction, datacube=True, matchup_number=matchup_number)

                print("[INFO] AERONET-OC matchup for L1c file")
                satobj = Hypso(path=l1c_nc_path, verbose=True)
                matchup_hypso_data = aeronet_oc_matchup_load_hypso_data(satobj, matchup, atmospheric_correction=atmospheric_correction, n_size=5)
                matchup_data = matchup_hypso_data | matchup_aeronet_data
                write_aeronet_oc_matchup_nc_file(satobj, matchup_data, atmospheric_correction=atmospheric_correction, datacube=True, matchup_number=matchup_number)

        except Exception as ex:
            print("[ERROR] Matchup failed!")
            print(ex)


    if TOGGLE_AERONET_OC_MATCHUPS_2:

        print("[INFO] Entering AERONET-OC matchups code")


        from hypso.aeronet_oc import aeronet_oc_detect_matchups, \
                                    aeronet_oc_generate_matchup, \
                                    aeronet_oc_matchup_load_hypso_data, \
                                    process_aeronet, \
                                    build_aeronet_queries, \
                                    process_hypso, \
                                    match_hypso_data, \
                                    get_column_prods
        
        from hypso.write import write_aeronet_oc_matchup_nc_file

        all_dfs = []

        matching_files = find_matching_files(HYPSO_DATA_DIR, coeff_type="moved", product_level="l2a", atmospheric_correction="polymer")

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

                    # Pull out unique days
                    unique_days = aoc_cb["aoc_datetime"].dt.date.unique()
                    unique_days_str = [day.strftime('%Y-%m-%d') for day in unique_days]


                    #print(aoc_lat)
                    #print(aoc_lon)
                    #print(unique_days)
                    #print(unique_days_str)


                    hypso_cb = process_hypso(satobj, aoc_lat, aoc_lon, atmospheric_correction="polymer")
                    #sat_cb = process_satellite(start_date="2024-06-01", end_date="2024-07-31",
                    #                latitude=aoc_lat, longitude=aoc_lon, sat="PACE",
                    #                selected_dates=unique_days_str)

                    matchups = match_hypso_data(hypso_cb, aoc_cb, cv_max=0.60, senz_max=60.0, 
                        min_percent_valid=55.0, max_time_diff=180, std_max=3)

                    all_dfs.append(matchups)

                    try:
                        matchups_df = pd.concat(all_dfs, ignore_index=True)
                        print(matchups_df.head())
                    except ValueError:
                        print("No rows in dataframe!")


                    dict_aoc = get_column_prods(matchups, "aoc")
                    waves_aoc = np.array(dict_aoc["rrs"]["wavelengths"])
                    rrs_aoc = matchups[dict_aoc["rrs"]["columns"]].to_numpy()

                    dict_hypso = get_column_prods(matchups, "hypso")
                    waves_hypso = np.array(dict_hypso["rrs"]["wavelengths"])
                    rrs_hypso = matchups[dict_hypso["rrs"]["columns"]].to_numpy()




            except Exception as ex:
                print(ex)


        try:
            matchups_df = pd.concat(all_dfs, ignore_index=True)
            print(matchups_df)
        except ValueError:
            print("No rows in dataframe!")

        matchups_df.to_parquet('aeronet.parquet', index=False)
        
        
        print("Done!")
                
            

    

   











if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        print("Usage: python process_l1d_dir.py <nc_dir_path>")
        
        if TEST_DIR is not None:
            print("Attempting to use test dir")
            dir_path = TEST_DIR
        else:
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


