"""This script finds the best point for Hypso and PACE data to match the RadCalNet site.
It also saves the pace spectrum at the point to a file. To future computation."""

# %% imports

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_map_utils.core.north_arrow import NorthArrow
import math


sys.path.insert(0, "/home/ariaa/smallSatLab/hypso-package/hypso/")
from hypso.hypso1 import Hypso1
from hypso.hypso2 import Hypso2
from hypso.spectral_analysis import get_closest_wavelength_index
from hypso.write import write_l1b_nc_file, write_l1c_nc_file, write_l1d_nc_file
from netCDF4 import Dataset
from pyresample import load_area

sys.path.insert(0, "/home/ariaa/smallSatLab/hypso-package/")
import os

from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from utils.write_l1d_from_file import write_l1d_from_file

# %% parameters

site = "gobabeb"  # 'gobabeb' or 'lacrau'
# site = 'lacrau'  # 'gobabeb' or 'lacrau
# site = "aeronet" # aeronet site

satellite = "h1"  # 'h1' or 'h2' (does not matter for lacrau)

make_l1d = False  # make l1d.nc files
do_hypso = True  # find best point for hypso data
do_pace = True  # find best point for pace data
save_pace = False  # save only spectrum at point to file


# %% set ideal lat/lon for the site

if site == "lacrau":
    ideal_lat = 43.558889
    ideal_lon = 4.864167
elif site == "gobabeb":
    ideal_lat = -23.6002
    ideal_lon = 15.11956
elif site == "aeronet":
    ideal_lat = 45.31390 # fill in later
    ideal_lon = 12.50830 # fill in later

print(site)
print(f"Latitude: {ideal_lat}")
print(f"Longitude: {ideal_lon}")

# %% plotting functions
def calculate_north_angle(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    delta_lon = lon2 - lon1
    term1 = math.sin(delta_lon) * math.cos(lat2)
    term2 = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(delta_lon)
    bearing = math.degrees(math.atan2(term1, term2))
    bearing = (bearing + 360) % 360  # Normalize to 0-360°
    return bearing

# %% read desired hypso and radcalnet files
if make_l1d:
    # make sure to use CRS: EPSG:4326 - WGS 84
    print(" [INFO] - Make sure to use CRS: EPSG:4326 - WGS 84")

    if site == "lacrau":
        core_path_h1 = "/home/ariaa/smallSatLab/data/h1/lacrau_2024-12-26T10-24-27Z/lacrau_2024-12-26T10-24-27Z"
        core_path_h2 = "/home/ariaa/smallSatLab/data/h2/lacrau_2024-12-26T11-15-54Z/lacrau_2024-12-26T11-15-54Z"
    elif site == "gobabeb":
        core_path_h1 = "/home/ariaa/smallSatLab/data/h1/gobabeb_2025-03-13T08-52-07Z/gobabeb_2025-03-13T08-52-07Z"
        core_path_h2 = "/home/ariaa/smallSatLab/data/h2/gobabeb_2025-03-04T09-20-31Z/gobabeb_2025-03-04T09-20-31Z"

    l1a_nc_file_h1 = core_path_h1 + "-l1a.nc"
    points_path_h1 = core_path_h1 + "-bin3.points"

    l1a_nc_file_h2 = core_path_h2 + "-l1a.nc"
    points_path_h2 = core_path_h2 + "-bin3.points"

    write_l1d_from_file(l1a_nc_file_h1, points_path_h1, "hypso1", flip=True)
    write_l1d_from_file(l1a_nc_file_h2, points_path_h2, "hypso2", flip=True)

    satobj_h1 = Hypso1(core_path_h1 + "-l1d.nc")
    satobj_h2 = Hypso2(core_path_h2 + "-l1d.nc")

    l1d_cube_h1 = satobj_h1.l1d_cube
    l1d_cube_h2 = satobj_h2.l1d_cube
elif do_hypso:
    core_path = "/home/ariaa/smallSatLab/data/RadCalNet/radcal_output"
    hypso_files = []

    for root, dirs, files in os.walk(core_path):
        for file in files:
            if "radcal" not in file:
                full_path = os.path.join(root, file)
                hypso_files.append(full_path)

    # NOTE - we have restricted the file selection in this way because the old nc-file structure has changed with the newer nc-files. These old files are compatible with the hypso-package code the "aria" branch
    for file in hypso_files:
        if "h1" in file and "l1d" in file and site in file and 'old' in file and 'moved' not in file and 'adjusted' not in file:
            print(f"[INTO] - hypso filepath: {file}")
            satobj_h1 = Hypso1(file)
            l1d_cube_h1 = satobj_h1.l1d_cube
            h1_file = file
        elif "h2" in file and "l1d" in file and site in file and 'old' in file and 'moved' not in file and 'adjusted' not in file:
            print(f"[INTO] - hypso filepath: {file}")
            satobj_h2 = Hypso2(file)
            l1d_cube_h2 = satobj_h2.l1d_cube
            h2_file = file

# %% find best point and plot
if do_hypso:
    if satellite == "h1":
        satobj, l1d_cube = [satobj_h2, l1d_cube_h2]
    elif satellite == "h2":
        satobj, l1d_cube = [satobj_h2, l1d_cube_h2]
    # for satobj, l1d_cube in [[satobj_h1, l1d_cube_h1], [satobj_h2, l1d_cube_h2]]: for both h1 and h2 TODO maybe add this back later
    # calculate best lat/lon
    min_error = np.inf
    for i in range(l1d_cube.shape[0]):
        for j in range(l1d_cube.shape[1]):
            error = np.sqrt((satobj.latitudes_indirect[i, j] - ideal_lat)**2 + (
                satobj.longitudes_indirect[i, j] - ideal_lon
            )**2)
            if error < min_error:
                min_error = error
                y_point = i
                x_point = j

    if site == "gobabeb":
        # manually adjust the point for gobabeb
        y_point = y_point - 5
        x_point = x_point + 10

    lat = satobj.latitudes_indirect
    lon = satobj.longitudes_indirect

    print(lat[y_point, x_point])
    print(lon[y_point, x_point])

    print(y_point, x_point)

    best_rgb_fit = []
    for c in [630, 532, 465]:  # wavelengths of red, green, blue
        best_rgb_fit.append(np.argmin(np.abs(c - satobj.wavelengths)))


    plt.figure(figsize=(10, 10))
    if site == "lacrau":
        img = l1d_cube[:, :, best_rgb_fit] * 2.2
    elif site == "gobabeb":
        img = l1d_cube[:, :, best_rgb_fit] * 1.9
    elif site == "aeronet":
        img = l1d_cube[:, :, best_rgb_fit]
    plt.imshow(img)
    plt.plot(x_point, y_point, "ro", markersize=1)
    plt.gca().set_aspect(6)  # Stretch image in y-direction (aspect < 1 stretches y)

    # Add zoomed inset
    zoom_area = 20  # pixels around the point
    ax = plt.gca()
    axins = zoomed_inset_axes(
        ax, zoom=20, loc="lower right"
    )  # Change 'loc' as needed
    axins.imshow(img)
    axins.plot(x_point, y_point, "ro", markersize=5)
    x1, x2 = max(x_point - zoom_area, 0), min(x_point + zoom_area, img.shape[1])
    y1, y2 = max(y_point - zoom_area, 0), min(y_point + zoom_area, img.shape[0])
    axins.set_xlim(x1, x2)
    axins.set_ylim(y2, y1)  # y is inverted in imshow
    axins.set_xticks([])
    axins.set_yticks([])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.show()

    # plot cutout of area
    # we manually set the size of the cutout to be approximately the same as for PACE
    y_start, y_stop = y_point - (300//6), y_point + (300//6)
    x_start, x_stop = x_point - 300, x_point + 300

    # filp capture if site is lacrau
    if site == "lacrau":
        img_cutout = np.fliplr(img[y_start:y_stop, x_start:x_stop, :])
    else:
        img_cutout = img[y_start:y_stop, x_start:x_stop, :]

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img_cutout)
    ax.set_aspect(6)  # Stretch image in y-direction

    # mark the selected point in the cutout
    point_y = (y_stop - y_start - 1) - (y_point - y_start)
    point_x = x_point - x_start
    ax.scatter(point_x, point_y, color='red')

    ax.set_axis_off()
    # Add scale bar and north arrow (if lat/lon available)
    if satellite == 'h1':
        m_per_px = 54     # from GSD calculations
    else:
        m_per_px = 68
    north_angle = calculate_north_angle(lat[-1,0],lon[-1,0], lat[0,0], lon[0,0] )
    scalebar = ScaleBar(m_per_px, units='m', location='lower right',
                        scale_loc='bottom')
    northarrow = NorthArrow(location="upper right",
                            rotation={"degrees": north_angle},
                            scale=0.3)
    ax.add_artist(scalebar)
    ax.add_artist(northarrow)
    if satellite == 'h1':
        ax.text(
            0.02,
            0.02,
            "HYPSO-1",
            color='black',
            fontsize=16,
            bbox=dict(facecolor='white', edgecolor='none', alpha=1),
            ha='left',
            va='bottom',
            transform=ax.transAxes,
        )
    else:
        ax.text(
            0.02,
            0.02,
            "HYPSO-2",
            color='black',
            fontsize=16,
            bbox=dict(facecolor='white', edgecolor='none', alpha=1),
            ha='left',
            va='bottom',
            transform=ax.transAxes,
        )
    plt.savefig(f'../../output_figures/calibration_paper/{site}_{satellite}_cutout.png', bbox_inches='tight', pad_inches=0.02, dpi=600)
    plt.show()

# %% PACE functions
def make_rgb_composite():
    # find best bands for RGB
    r, g, b = 630, 532, 465

    rgb_bands = np.array(
        [
            np.argmin(np.abs(np.concatenate((blue_bands, red_bands)) - color))
            for color in [r, g, b]
        ]
    )

    num_blue_bands = l1b_capture_blue.shape[0]

    capture = np.zeros((l1b_capture_blue.shape[1], l1b_capture_blue.shape[2], 3))
    i = 0
    for c in rgb_bands:
        if c < num_blue_bands:
            capture[:, :, i] = l1b_capture_blue[c, :, :]
        else:
            capture[:, :, i] = l1b_capture_red[c - num_blue_bands, :, :]
        i += 1
    print(f"Capture shape: {capture.shape}")

    # clip, flip, and increase brightness on composite
    composite = 255 * capture / (capture.max())

    composite_bright = composite * 3
    composite_clipped = np.clip(composite_bright, 0, 255).astype("uint8")
    composite_flipped = np.flip(composite_clipped, axis=0)

    return composite_clipped


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


def merge_red_blue_bands(y_start, y_stop, x_start, x_stop):
    overlaps_blue = []
    overlaps_red = []
    for i in range(len(blue_bands)):
        if (blue_bands[i] > red_bands).any():
            idx = int(np.argmin(np.abs(red_bands - blue_bands[i])))
            overlaps_blue.append(i)
            overlaps_red.append(idx)

    l1b_capture_blue_reduced = l1b_capture_blue[:, y_start:y_stop, x_start:x_stop]
    l1b_capture_red_reduced = l1b_capture_red[:, y_start:y_stop, x_start:x_stop]

    l_b, l_r, l_o = len(blue_bands), len(red_bands), len(overlaps_blue)

    l1b_merged = np.zeros(
        (
            l_b + l_r - l_o,
            l1b_capture_blue_reduced.shape[1],
            l1b_capture_blue_reduced.shape[2],
        )
    )

    l1b_merged[: l_b - l_o, :, :] = l1b_capture_blue_reduced[:-l_o, :, :]
    l1b_merged[l_b - l_o : l_b, :, :] = 0.5 * (
        l1b_capture_blue_reduced[overlaps_blue, :, :]
        + l1b_capture_red_reduced[overlaps_red, :, :]
    )
    l1b_merged[l_b:, :, :] = l1b_capture_red_reduced[l_o:, :, :]

    # make a merged bands array
    bands_merged = np.zeros(l_b + l_r - l_o)
    bands_merged[: l_b - l_o] = blue_bands[:-l_o]
    bands_merged[l_b - l_o : l_b] = 0.5 * (
        blue_bands[overlaps_blue] + red_bands[overlaps_red]
    )  # make the middle bands the average of the blue and red bands
    bands_merged[l_b:] = red_bands[l_o:]

    # print(l1b_merged.shape)
    return l1b_merged, bands_merged


# %% Find best point for PACE data, and save
if do_pace:
    # specify the path to your netCDF file
    core_path = "/home/ariaa/smallSatLab/data/PACE"
    if site == "lacrau":
        l1b_nc_file = os.path.join(
            core_path, "lacrau_2024-12-26T11-59-01Z_h1_h2/PACE_OCI.20241226T115901.L1B.V3.nc"
        )
    elif site == "gobabeb":
        if satellite == "h1":
            l1b_nc_file = os.path.join(
                core_path,
                "gobabeb_2025-03-13T12-21-53Z_h1/PACE_OCI.20250313T122153.L1B.V3.nc",
            )
        elif satellite == "h2":
            l1b_nc_file = os.path.join(
                core_path,
                "gobabeb_2025-03-04T09-20-31Z_h2/PACE_OCI.20250304T120217.L1B.V3.nc",
            )
        # '/home/ariaa/smallSatLab/data/h2/gobabeb_2025-03-04T09-20-31Z/gobabeb_2025-03-04T09-20-31Z'
    elif site == "aeronet":
        if satellite == "h1":
            l1b_nc_file = os.path.join(
                core_path,
                "aeronetvenice_2025-05-12Z_h1/PACE_OCI.20250512T115644.L1B.V3.nc",
            )
        elif satellite == "h2":
            l1b_nc_file = os.path.join(
                core_path,
                "aeronetvenice_2025-05-14Z_h2/PACE_OCI.20250514T112858.L1B.V3.nc",
            )
    else:
        raise ValueError("Invalid site. Choose 'lacrau' or 'gobabeb'.")

    l1b_nc = Dataset(l1b_nc_file, mode="r")

    # load lat/lon data
    lat = l1b_nc.groups["geolocation_data"].variables["latitude"][:]
    lon = l1b_nc.groups["geolocation_data"].variables["longitude"][:]

    # load data
    l1b_capture_blue = l1b_nc.groups["observation_data"].variables["rhot_blue"][:]
    l1b_capture_red = l1b_nc.groups["observation_data"].variables["rhot_red"][:]

    # load wavelengths
    blue_bands = np.array(l1b_nc.groups["sensor_band_parameters"]["blue_wavelength"][:])
    red_bands = np.array(l1b_nc.groups["sensor_band_parameters"]["red_wavelength"][:])

    # make rgb composite
    composite_clipped = make_rgb_composite()

    # calculate best matching point
    y_point, x_point, y_start, y_stop, x_start, x_stop = find_point_from_latlon()

    # manually adjust the point for gobabeb
    if site == "gobabeb":
        y_point = y_point + 1

    # merge blue and red bands
    l1b_merged, bands_merged = merge_red_blue_bands(y_start, y_stop, x_start, x_stop)

    # plot spectrum at point
    plt.figure(figsize=(12, 6))
    plt.plot(bands_merged, l1b_merged[:, y_point - y_start, x_point - x_start])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("ToA Reflectance")
    plt.title(f"PACE - {site} - {lat[y_point, x_point]} - {lon[y_point, x_point]}")
    plt.grid(True)
    plt.show()


    # plot cutout of area

    # flip capture and lat/lon
    composite_flipped = (np.flipud(composite_clipped[y_start:y_stop, x_start:x_stop, :]))
    lat_flipped = (np.flipud(lat))
    lon_flipped = (np.flipud(lon))

    if site == "lacrau":
        composite_flipped = (composite_flipped * 1.5).astype('uint8')
    elif site == "gobabeb":
        composite_flipped = (composite_flipped * 0.9).astype('uint8')

    # plot image of area
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(composite_flipped)

    # mark the selected point in the cutout
    point_y = (y_stop - y_start - 1) - (y_point - y_start)
    point_x = x_point - x_start
    ax.scatter(point_x, point_y, color='red')

    ax.set_axis_off()

    # Add scale bar and north arrow (if lat/lon available)
    km_per_px = 1
    north_angle = calculate_north_angle(lat_flipped[-1,0],lon_flipped[-1,0], lat_flipped[0,0], lon_flipped[0,0] )
    scalebar = ScaleBar(km_per_px, units='km', location='lower right',
                        scale_loc='bottom')
    northarrow = NorthArrow(location="upper right",
                            rotation={"degrees": north_angle},
                            scale=0.3)
    ax.add_artist(scalebar)
    ax.add_artist(northarrow)
    ax.text(
        0.02,
        0.02,
        "PACE",
        color='black',
        fontsize=16,
        bbox=dict(facecolor='white', edgecolor='none', alpha=1),
        ha='left',
        va='bottom',
        transform=ax.transAxes,
    )


    plt.savefig(f'../../output_figures/calibration_paper/{site}_PACE_cutout.png', bbox_inches='tight', pad_inches=0.02, dpi=600)
    plt.show()

    # save pace spectrum
    if save_pace:
        if site == "lacrau":
            np.save(
                os.path.join(
                    core_path,
                    f"lacrau_2024-12-26T11-59-01Z/PACE_spectrum_lat{int(ideal_lat)}_lon{int(ideal_lon)}_2024-12-26T11-59-01.npy",
                ),
                l1b_merged[:, y_point - y_start, x_point - x_start],
            )
            print(f"Saved PACE spectrum for {site}")
            np.save(
                os.path.join(
                    core_path,
                    f"lacrau_2024-12-26T11-59-01Z/PACE_bands_lat{int(ideal_lat)}_lon{int(ideal_lon)}_2024-12-26T11-59-01.npy",
                ),
                bands_merged,
            )
            print(f"Saved PACE bands for {site}")
        elif site == "gobabeb":
            if satellite == "h1":
                np.save(
                    os.path.join(
                        core_path,
                        f"gobabeb_2025-03-13T12-21-53Z_h1/PACE_spectrum_lat{int(ideal_lat)}_lon{int(ideal_lon)}_2025-03-13T12-21-53.npy",
                    ),
                    l1b_merged[:, y_point - y_start, x_point - x_start],
                )
                np.save(
                    os.path.join(
                        core_path,
                        f"gobabeb_2025-03-13T12-21-53Z_h1/PACE_bands_lat{int(ideal_lat)}_lon{int(ideal_lon)}_2025-03-13T12-21-53.npy",
                    ),
                    bands_merged,
                )
            elif satellite == "h2":
                np.save(
                    os.path.join(
                        core_path,
                        f"gobabeb_2025-03-04T09-20-31Z_h2/PACE_spectrum_lat{int(ideal_lat)}_lon{int(ideal_lon)}_2025-03-04T09-20-31.npy",
                    ),
                    l1b_merged[:, y_point - y_start, x_point - x_start],
                )
                np.save(
                    os.path.join(
                        core_path,
                        f"gobabeb_2025-03-04T09-20-31Z_h2/PACE_bands_lat{int(ideal_lat)}_lon{int(ideal_lon)}_2025-03-04T09-20-31.npy",
                    ),
                    bands_merged,
                )
            print(f"Saved PACE spectrum for {site}")
            print(f"Saved PACE bands for {site}")
        elif site == "aeronet":
            if satellite == "h1":
                np.save(
                    os.path.join(
                        core_path,
                        f"aeronetvenice_2025-05-12Z_h1/PACE_spectrum_lat{int(ideal_lat)}_lon{int(ideal_lon)}_2025-05-12T11-56-44.npy",
                    ),
                    l1b_merged[:, y_point - y_start, x_point - x_start],
                )
                np.save(
                    os.path.join(
                        core_path,
                        f"aeronetvenice_2025-05-12Z_h1/PACE_bands_lat{int(ideal_lat)}_lon{int(ideal_lon)}_2025-05-12T11-26-44.npy",
                    ),
                    bands_merged,
                )
            elif satellite == "h2":
                np.save(
                    os.path.join(
                        core_path,
                        f"aeronetvenice_2025-05-14Z_h2/PACE_spectrum_lat{int(ideal_lat)}_lon{int(ideal_lon)}_2025-05-14T11-28-58.npy",
                    ),
                    l1b_merged[:, y_point - y_start, x_point - x_start],
                )
                np.save(
                    os.path.join(
                        core_path,
                        f"aeronetvenice_2025-05-14Z_h2/PACE_bands_lat{int(ideal_lat)}_lon{int(ideal_lon)}_2025-05-14T11-28-58.npy",
                    ),
                    bands_merged,
                )
            print(f"Saved PACE spectrum for {site}")
            print(f"Saved PACE bands for {site}")
        else:
            raise ValueError("Invalid site. Choose 'lacrau' or 'gobabeb'.")
