#!/usr/bin/env python3

import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/camerop/ARIEL/atmospheric_correction/hypso-package/hypso')
sys.path.insert(0, '/home/camerop/ARIEL/atmospheric_correction/hypso-package/hypso1_calibration')
sys.path.insert(0, '/home/camerop/ARIEL/atmospheric_correction/hypso-package/hypso2_calibration')

from hypso import Hypso
from hypso.write import write_l1b_nc_file, write_l1c_nc_file, write_l1d_nc_file, write_l2a_nc_file, write_products_nc_file

TEST_DIR = "/home/cameron/Nedlastinger/image64N9E_2025-04-23T10-28-01Z"

RAD_CAL_COEFFS = "moved"

TOGGLE_OCSMART = False
TOGGLE_ACOLITE = False
TOGGLE_6SV1 = False
TOGGLE_SREM = False
TOGGLE_POLYMER = True

TOGGLE_RUN_AC = True
TOGGLE_READ_AC = True


POLYMER_PATH = "/home/camerop/ARIEL/atmospheric_correction/Polymer/polymer/"
EOREAD_PATH = "/home/camerop/ARIEL/atmospheric_correction/Polymer/eoread/"
EOTOOLS_PATH = "/home/camerop/ARIEL/atmospheric_correction/Polymer/eotools/"

OCSMART_PATH = ""
ACOLITE_PATH = ""

DEM_PATH = ""

EARTHDATA_u = "cpenne"
EARTHDATA_p = "Dec1!onJG0@1LogoMen5un!"


def main(l1a_nc_path, lats_path=None, lons_path=None, coeff_type=None):
    # Check if the first file exists
    if not os.path.isfile(l1a_nc_path):
        print(f"Error: The file '{l1a_nc_path}' does not exist.")
        return

    # Process the first file
    print(f"Processing file: {l1a_nc_path}")

    nc_file = Path(l1a_nc_path)

    satobj = Hypso(path=nc_file, verbose=True)

    #print(satobj.nc_attrs['target_latitude'])
    #print(satobj.nc_attrs['target_longitude'])

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

                satobj.generate_l1c_cube()
                satobj.generate_l1d_cube(use_direct_georef=False)

            except Exception as ex:
                print(ex)
                print('Indirect georeferencing has failed. Defaulting to direct georeferencing.')

                satobj.run_direct_georeferencing()
                satobj.generate_l1b_cube(coeff_type=coeff_type)
                #satobj.generate_l1c_cube()
                satobj.generate_l1d_cube(use_direct_georef=True)

        else:
            satobj.run_direct_georeferencing()

            satobj.generate_l1b_cube(coeff_type=coeff_type)
            #satobj.generate_l1c_cube()
            satobj.generate_l1d_cube(use_direct_georef=True)

        datacube = False

        write_l1b_nc_file(satobj, overwrite=True, datacube=datacube) 
        #write_l1c_nc_file(satobj, overwrite=True, datacube=datacube)
        write_l1d_nc_file(satobj, overwrite=True, datacube=datacube)


    # Atmospheric correction

    if TOGGLE_OCSMART:
        satobj.ocsmart_dir = "/home/_shared/ARIEL/atmospheric_correction/OC-SMART/OC-SMART_with_HYPSO_9-29-25_release/"
        if TOGGLE_RUN_AC:
            satobj.ac_ocsmart_stage_input()
            satobj.ac_ocsmart_run_correction()
        if TOGGLE_READ_AC:
            satobj.ac_ocsmart_open_output()
            write_l2a_nc_file(satobj=satobj, correction="ocsmart", overwrite=True, datacube=False)

    if TOGGLE_ACOLITE:
        satobj.acolite_dir = "/home/_shared/ARIEL/atmospheric_correction/acolite/"
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
            satobj.ac_polymer_run_correction(polymer_path=POLYMER_PATH, eoread_path=EOREAD_PATH, eotools_path=EOTOOLS_PATH)
        if TOGGLE_READ_AC:
            datasets = satobj.ac_polymer_open_output()
            #satobj.products['logchl'] = datasets['logchl']

            write_l2a_nc_file(satobj=satobj, correction="polymer", overwrite=True, datacube=False)
            write_products_nc_file(satobj, overwrite=True, file_name="polymer_chl.nc")





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

    #dir_path = "/home/cameron/Nedlastinger/frohavet_2025-05-22T11-20-44Z"
    #dir_path = "/home/cameron/Nedlastinger/image63N9E_2025-05-11T10-04-27Z"
    
    #dir_path = "/home/cameron/Nedlastinger/image64N9E_2025-04-23T10-24-32Z"
    #dir_path = "/home/cameron/Nedlastinger/image64N9E_2025-04-23T10-28-01Z"

    base_path = dir_path.rstrip('/')

    folder_name = os.path.basename(base_path)
    l1a_nc_path = os.path.join(base_path, f"{folder_name}-l1a.nc")
    l1d_nc_path = os.path.join(base_path, f"{folder_name}-l1d.nc")
    lats_path = os.path.join(base_path, "processing-temp", "latitudes_indirectgeoref.dat")
    lons_path = os.path.join(base_path, "processing-temp", "longitudes_indirectgeoref.dat") 


    #print(base_path)
    #print(folder_name)
    #print(lat_file)
    #print(lon_file)
    #lats_path = sys.argv[2] if len(sys.argv) == 4 else None
    #lons_path = sys.argv[3] if len(sys.argv) == 4 else None

    main(l1a_nc_path, lats_path, lons_path, coeff_type=RAD_CAL_COEFFS)

    #dst_dir = "/home/_shared/ARIEL/atmospheric_correction/OC-SMART/OC-SMART_with_HYPSO/L1B/"
    #dst_file = os.path.join(dst_dir, "HYPSO2_HSI_" + str(folder_name) + "-l1d.nc")

    #import shutil
    #shutil.copy2(l1d_nc_path, dst_file)

