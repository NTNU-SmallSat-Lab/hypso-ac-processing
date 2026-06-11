#!/usr/bin/env python3

import os
import sys
import numpy as np
from pathlib import Path
import pandas as pd

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
TOGGLE_AERONET_OC_MATCHUPS_2 = False

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


def main(l1a_nc_path, l1b_nc_path, lats_path=None, lons_path=None, coeff_type=None):

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
                    """
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
                    """
                    satobj.generate_l1b_cube()
                    #satobj.generate_l1c_cube()
                    #satobj.generate_l1d_cube(use_direct_georef=False, generate_figures=GENERATE_FIGURES)

                except Exception as ex:
                    print(ex)
                    print('Indirect georeferencing has failed. Defaulting to direct georeferencing.')

                    #satobj.run_direct_georeferencing()
                    satobj.generate_l1b_cube(coeff_type=coeff_type)
                    #satobj.generate_l1c_cube()
                    #satobj.generate_l1d_cube(use_direct_georef=True, generate_figures=GENERATE_FIGURES)

            else:
                #satobj.run_direct_georeferencing()

                satobj.generate_l1b_cube(coeff_type=coeff_type)
                #satobj.generate_l1c_cube()
                #satobj.generate_l1d_cube(use_direct_georef=True, generate_figures=GENERATE_FIGURES)


            #from hypso.reflectance import compute_csiro_srfs

            #satobj._get_fwhm_unbinned()

            satobj.compute_csiro_srfs()

            #sensor_wavelengths_unbinned = satobj.wavelengths_unbinned
            #sensor_wavelengths = satobj.wavelengths
            #sensor_fwhm = satobj.fwhm_unbinned
            #bin_factor = satobj.bin_factor


            #srfs_csr, ssi, solar_wavelengths, binned_srfs, effective_fwhm, esun = compute_csiro_srfs(sensor_wavelengths_unbinned,
            #            sensor_fwhm,
            #            bin_factor,
            #            generate_figures= False
            #            )


            print(satobj.csiro_srfs_csr) 
            print(satobj.csiro_ssi)
            print(satobj.csiro_solar_wavelengths)
            print(satobj.csiro_binned_srfs)
            print(satobj.csiro_effective_fwhm)
            print(satobj.csiro_esun)


            exit()
            


            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.ylim(0, 2500)
            plt.plot(sensor_wavelengths, esun, label='esun', color='blue', linewidth=2)
            plt.savefig('plot.png')            

            plt.figure(figsize=(10, 6))
            plt.ylim(0, 1)
            plt.plot(solar_wavelengths, binned_srfs[40,:], label='srf', color='blue', linewidth=2)
            plt.savefig('plot_srf.png')            


    else:

        # Check if the first file exists
        if not os.path.isfile(l1d_nc_path):
            print(f"Error: The file '{l1d_nc_path}' does not exist.")
            return

        # Process the first file
        print(f"Processing file: {l1d_nc_path}")

        nc_file = Path(l1d_nc_path)

        satobj = Hypso(path=nc_file, verbose=True, label=RAD_CAL_COEFFS)

 



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


