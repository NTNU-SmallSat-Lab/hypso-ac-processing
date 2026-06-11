

import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso')
sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso1_calibration')
sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso2_calibration')

from hypso import Hypso
from hypso.write import write_l1b_nc_file, write_l1c_nc_file, write_l1d_nc_file, write_l2a_nc_file, write_products_nc_file
from hypso.classification import decode_jon_cnn_labels, decode_jon_cnn_cloud_mask, decode_jon_cnn_water_mask, decode_jon_cnn_land_mask

import pandas as pd
import requests
from io import StringIO
from datetime import datetime, timedelta
import re

aeronet_oc_data_dir = "/home/cameron/Nedlastinger/AERONET_OC_Data"
ac_algorithm = "polymer"
aeronet_oc_sites_csv_path = "/home/cameron/Nedlastinger/AERONET_OC_Sites.csv"
l2a_nc_path = "/home/cameron/Nedlastinger/aeronetvenice_2025-06-12T09-58-02Z-moved-l2a-polymer.nc"


# Load the CSV file
df = pd.read_csv(aeronet_oc_sites_csv_path)  # Replace with your actual file path
print(df.columns.tolist())




if not os.path.isfile(l2a_nc_path):
    print(f"Error: The file '{l2a_nc_path}' does not exist.")
    exit()

# Process the first file
print(f"Processing file: {l2a_nc_path}")

nc_file = Path(l2a_nc_path)

satobj = Hypso(path=nc_file, verbose=True)


capture_target = str(satobj.capture_target).lower()

print("Searching AERONET-OC site matchup for " + capture_target + " HYPSO target")

matching_rows = df[df['HYPSO_NAME'] == capture_target]

if not matching_rows.empty:
    # Get the first matching row
    row = matching_rows.iloc[0]

    hypsos_name = row.HYPSO_NAME  # Note: spaces become underscores
    aeronet_name = row.AERONETOC_NAME
    aeronet_latitude = row.LATITUDE
    aeronet_longitude = row.LONGITUDE
    elevation = row.ELEVATION
    
    print("Detected AERONET-OC site match:")
    print(f"{hypsos_name} - {aeronet_name}: ({aeronet_latitude}, {aeronet_longitude})")
else:
    exit()


hypso_latitudes = satobj.latitudes
hypso_longitudes = satobj.longitudes

capture_shape = satobj.l2a_cube[ac_algorithm].shape[0:2]
print(capture_shape)





min_error = np.inf
for i in range(capture_shape[0]):
    for j in range(capture_shape[1]):
        error = np.abs(hypso_latitudes[i, j] - aeronet_latitude) + np.abs(hypso_longitudes[i, j] - aeronet_longitude)
        if error < min_error:
            min_error = error
            y_point = i
            x_point = j

print(f"Closest (lat,lon): ({hypso_latitudes[y_point, x_point]}, {hypso_longitudes[y_point, x_point]})")
print(f"Coordinates (y,x): ({y_point}, {x_point})")





# https://aeronet.gsfc.nasa.gov/print_web_data_help_v3_seaprism_new.html
'''
def download_lwn_for_site(site_name, year, month, day, data_level=1.0, output_filename=None):
    """Download Lwn data for a single site and date"""
    data_type = 'LWN10' if data_level == 1.0 else 'LWN15'
    
    url = (f"https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3"
           f"?site={site_name}&year={year}&month={month}&day={day}"
           f"&year2={year}&month2={month}&day2={day}"
           f"&{data_type}=1&AVG=10&if_no_html=1")
    
    response = requests.get(url, verify=False)
    

    if response.status_code == 200:
        # Save raw response text to file
        with open(output_filename, 'w') as f:
            f.write(response.text)
        print(f"Saved response to: {output_filename}")
        return True
    else:
        print(f"Error: {response.status_code}")
        return False

    #if response.status_code == 200:
    #    df = pd.read_csv(StringIO(response.text), delimiter=',', comment='#')
    #    return df
    #else:
    #    print(f"Failed for {site_name}: HTTP {response.status_code}")
    #    return None
'''




def download_aeronet_oc_lwn_data(site_name, year, month, day, data_level=1.0, output_dir='aeronet_data'):
    """
    Download Lwn data for a single site and date.
    Skips download if file already exists.
    
    Parameters:
    -----------
    site_name : str
        AERONET site name
    year, month, day : int
        Date for download
    data_level : float
        1.0 or 1.5
    output_dir : str
        Base directory for storing files
    
    Returns:
    --------
    str or None
        Path to the downloaded file if successful, None if failed
    """
    
    # Create subdirectory for this site
    site_dir = Path(output_dir) / site_name
    site_dir.mkdir(parents=True, exist_ok=True)
    
    # Build filename
    data_type = 'LWN10' if data_level == 1.0 else 'LWN15'
    filename = f"{site_name}_{data_type}_{year}{month:02d}{day:02d}.csv"
    filepath = site_dir / filename
    
    # Check if file already exists
    if filepath.exists():
        print(f"File already exists: {filepath}")
        return str(filepath)
    
    # Build URL
    url = (f"https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3"
           f"?site={site_name}&year={year}&month={month}&day={day}"
           f"&year2={year}&month2={month}&day2={day}"
           f"&{data_type}=1&AVG=10&if_no_html=1")
    
    print(f"Downloading: {site_name} for {year}-{month:02d}-{day:02d}")
    
    # Download
    response = requests.get(url, verify=False)
    
    if response.status_code == 200:
        # Save raw response text to file
        with open(filepath, 'w') as f:
            f.write(response.text)
        print(f"Saved to: {filepath}")
        return str(filepath)
    else:
        print(f"Error for {site_name}: HTTP {response.status_code}")
        return None


def read_aeronet_oc_lwn_file(filepath):
    """
    Read AERONET CSV file, skipping the first 5 metadata lines.
    Line 6 (index 5) contains the column headers.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip first 5 metadata lines, keep from line 6 onwards
    data_lines = lines[5:]  # lines[5] is the 6th line (column headers)
    data_text = ''.join(data_lines)
    
    # Read into dataframe
    df = pd.read_csv(StringIO(data_text), delimiter=',')
    
    return df


def find_closest_aeronet_time(df, dt, time_column="Time(hh:mm:ss)"):
    """
    Find the row in AERONET DataFrame with time closest to given datetime.
    
    Parameters:
    -----------
    df : pandas DataFrame
        AERONET data with 'Time' column in format 'HH:MM:SS'
    dt : datetime
        Reference datetime (from satellite)
    
    Returns:
    --------
    row : pandas Series
        The closest matching row
    """
    # Extract time from datetime
    target_time = dt.time()
    
    # Parse AERONET time strings to datetime.time objects
    df['parsed_time'] = pd.to_datetime(df[time_column], format='%H:%M:%S').dt.time
    
    # Calculate time difference (convert to minutes for easier comparison)
    def time_diff(time_obj):
        # Convert time to minutes since midnight
        target_minutes = target_time.hour * 60 + target_time.minute + target_time.second / 60
        row_minutes = time_obj.hour * 60 + time_obj.minute + time_obj.second / 60
        
        # Circular time difference (handles wrap around midnight)
        diff = abs(row_minutes - target_minutes)
        diff = min(diff, 1440 - diff)  # 1440 minutes in a day
        return diff
    
    df['time_diff_minutes'] = df['parsed_time'].apply(time_diff)
    
    # Find row with minimum time difference
    closest_idx = df['time_diff_minutes'].idxmin()
    closest_row = df.loc[closest_idx]
    
    print(f"Target time: {target_time}")
    print(f"Closest AERONET time: {closest_row[time_column]}")
    print(f"Difference: {closest_row['time_diff_minutes']:.2f} minutes")
    
    return closest_row




def extract_aeronet_products(closest_series):
    """
    Extract and organize all AERONET-OC products from a pandas Series (row).
    
    Parameters:
    -----------
    closest_series : pandas Series
        A single row from AERONET-OC DataFrame (result of .iloc[] or .loc[])
    
    Returns:
    --------
    dict : Dictionary containing categorized products by type and wavelength
    """
    
    products = {
        'Lw': {},
        'Lt': {},
        'Lwn': {},
        'Lwn_fQ': {},
        'Exact_Wavelengths': {}
    }
    
    # Iterate through all columns in the Series
    for col_name in closest_series.index:
        print(col_name)
        
        # Parse Lw[412], Lw[443], etc.
        if col_name.startswith('Lwn[') and col_name.endswith(']'):

            print("Match found")    

            value = closest_series[col_name]
            match = re.search(r'\[(\d+)nm\]', col_name)
            if match:
                wavelength = int(match.group(1))

            print(value)
            print(wavelength)

            products['Lwn'][wavelength] = value
        
    
        # Parse Lw_f/Q[412], Lw_f/Q[443], etc.
        if col_name.startswith('Lw_f/Q[') and col_name.endswith(']'):

            print("Match found")    

            value = closest_series[col_name]
            match = re.search(r'\[(\d+)nm\]', col_name)
            if match:
                wavelength = int(match.group(1))

            print(value)
            print(wavelength)

            products['Lwn_fQ'][wavelength] = value


        # Parse Exact_Wavelengths(um)_412, etc.
        if col_name.startswith('Exact_Wavelengths(um)_'):

            print("Match found")    

            value = closest_series[col_name]
            try:
                wavelength = int(col_name.split('_')[-1])
            except:
                break

            print(value)
            print(wavelength)

            products['Exact_Wavelengths'][wavelength] = value


    return products

dt = satobj.capture_datetime

output_file = f"aeronet_{dt.year}{dt.month:02d}{dt.day:02d}.csv"

aeronet_oc_lwn_csv_file = download_aeronet_oc_lwn_data(
                        site_name=aeronet_name,  
                        year=dt.year,
                        month=dt.month,
                        day=dt.day,
                        data_level=1.0,
                        output_dir=aeronet_oc_data_dir
                    )


if aeronet_oc_lwn_csv_file is not None:
    aeronet_data_df = read_aeronet_oc_lwn_file(aeronet_oc_lwn_csv_file)

aeronet_data_df = find_closest_aeronet_time(aeronet_data_df, dt)


print(aeronet_data_df)
print(type(aeronet_data_df))


products = extract_aeronet_products(aeronet_data_df)

print(products)



lwn_data = products["Lwn"]

lwn_data = {wl: val for wl, val in products["Lwn"].items() if val != -999}

import matplotlib.pyplot as plt
wavelengths = sorted(lwn_data.keys())
values = [lwn_data[wl] for wl in wavelengths]

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, values, 'o-', linewidth=2, markersize=8, color='blue')

# Labels and title
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Lwn (mW cm⁻² μm⁻¹ sr⁻¹)', fontsize=12)
plt.title('AERONET-OC Lwn Spectrum', fontsize=14)

# Add grid
plt.grid(True, alpha=0.3)

# Optionally add value labels on points
for wl, val in zip(wavelengths, values):
    plt.annotate(f'{val:.4f}', (wl, val), textcoords="offset points", 
                 xytext=(0, 10), ha='center', fontsize=9)

plt.tight_layout()
plt.show()





exit()











'''
def find_point_from_latlon():
    # calculate best matching point
    min_error = np.inf
    for i in range(composite_clipped.shape[0]):
        for j in range(composite_clipped.shape[1]):
            error = np.abs(lat[i, j] - ideal_lat) + np.abs(lon[i, j] - ideal_lon)
            if error < min_error:
                min_error = error
                y_point = i
                x_point = j

    print(f"Closest (lat,lon): ({lat[y_point, x_point]}, {lon[y_point, x_point]})")
    print(f"Coordinates (y,x): ({y_point}, {x_point})")

    plt.figure(figsize=(12, 12))
    plt.imshow(composite_clipped)
    plt.plot(x_point, y_point, "ro", markersize=1)

    # Add zoomed inset
    zoom_area = 35  # pixels around the point
    ax = plt.gca()
    axins = zoomed_inset_axes(ax, zoom=10, loc="upper left")  # Change 'loc' as needed
    axins.imshow(composite_clipped)
    axins.plot(x_point, y_point, "ro", markersize=2)
    x_start, x_stop = (
        max(x_point - zoom_area, 0),
        min(x_point + zoom_area, composite_clipped.shape[1]),
    )
    y_start, y_stop = (
        max(y_point - zoom_area, 0),
        min(y_point + zoom_area, composite_clipped.shape[0]),
    )
    axins.set_xlim(x_start, x_stop)
    axins.set_ylim(y_start, y_stop)  # y is inverted in imshow
    axins.set_xticks([])
    axins.set_yticks([])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.show()
    return y_point, x_point, y_start, y_stop, x_start, x_stop
'''