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

import matplotlib.pyplot as plt

from hypso import Hypso
from hypso.write import write_l1b_nc_file, write_l1c_nc_file, write_l1d_nc_file, write_l2a_nc_file, write_products_nc_file
from hypso.classification import decode_jon_cnn_labels, decode_jon_cnn_cloud_mask, decode_jon_cnn_water_mask, decode_jon_cnn_land_mask

#from hypso.aeronet_oc import build_aeronet_queries, format_capture_date



RAD_CAL_COEFFS = "moved"



import pysptools





#
#------------------------------------------------------------------------------
# Copyright (c) 2013-2014, Christian Therien
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#------------------------------------------------------------------------------
#
# vd.py - This file is part of the PySptools package.
#

"""
HfcVd function
"""

# Based on the noise estimation from Hyperspectral Subspace Identification José M. Bioucas-Dias, Member, IEEE, and José M. P. Nascimento, Member, IEEE



def est_noise(y, noise_type='additive'):
    """
    This function infers the noise in a
    hyperspectral data set, by assuming that the
    reflectance at a given band is well modelled
    by a linear regression on the remaining bands.

    Parameters:
        y: `numpy array`
            a HSI cube ((m*n) x p)

       noise_type: `string [optional 'additive'|'poisson']`

    Returns: `tuple numpy array, numpy array`
        * the noise estimates for every pixel (N x p)
        * the noise correlation matrix estimates (p x p)

    Copyright:
        Jose Nascimento (zen@isel.pt) and Jose Bioucas-Dias (bioucas@lx.it.pt)
        For any comments contact the authors
    """
    def est_additive_noise(r):
        small = 1e-6
        L, N = r.shape
        w=np.zeros((L,N), dtype=float)
        RR=np.dot(r,r.T)
        RRi = np.linalg.pinv(RR+small*np.eye(L))
        RRi = np.matrix(RRi)
        for i in range(L):
            XX = RRi - (RRi[:,i]*RRi[i,:]) / RRi[i,i]
            RRa = RR[:,i]
            RRa[i] = 0
            beta = np.dot(XX, RRa)
            beta[0,i]=0;
            w[i,:] = r[i,:] - np.dot(beta,r)
        Rw = np.diag(np.diag(np.dot(w,w.T) / N))
        return w, Rw

    y = y.T
    L, N = y.shape
    #verb = 'poisson'
    if noise_type == 'poisson':
        sqy = np.sqrt(y * (y > 0))
        u, Ru = est_additive_noise(sqy)
        x = (sqy - u)**2
        w = np.sqrt(x)*u*2
        Rw = np.dot(w,w.T) / N
    # additive
    else:
        w, Rw = est_additive_noise(y)
    return w.T, Rw.T








def process_datacube_for_noise_estimation(datacube):
    """
    Process a hyperspectral datacube for noise estimation:
    1. Remove bands that are all NaN
    2. Flatten to (m*n, p)
    3. Remove samples (pixels) that are all NaN
    4. Run noise estimation on valid samples
    5. Reconstruct noise matrix with valid samples in original positions
    
    Parameters:
    -----------
    datacube : numpy.ndarray
        3D array of shape (m, n, p) where p are spectral bands
    
    Returns:
    --------
    noise_cube : numpy.ndarray
        3D array of shape (m, n, p) with noise estimates
        NaN values remain for invalid pixels
    noise_estimates : numpy.ndarray
        Array of shape (valid_pixels, p) with noise estimates for valid pixels
    valid_positions : numpy.ndarray
        Array of shape (valid_pixels,) with the original flattened indices
    """
    m, n, p = datacube.shape
    total_pixels = m * n
    
    print("=" * 60)
    print("DATACUBE PROCESSING FOR NOISE ESTIMATION")
    print("=" * 60)
    print(f"Original shape: ({m}, {n}, {p})")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Total bands: {p}")
    print("-" * 60)
    
    # ============================================================
    # STEP 1: Remove bands that are ALL NaN
    # ============================================================
    print("\n[STEP 1] Removing all-NaN bands...")
    
    # Find bands that are all NaN
    all_nan_bands = []
    valid_bands = []
    
    for b in range(p):
        band_data = datacube[:, :, b]
        if np.isnan(band_data).all():
            all_nan_bands.append(b)
        else:
            valid_bands.append(b)
    
    print(f"  - Removed {len(all_nan_bands)} all-NaN bands")
    print(f"  - Kept {len(valid_bands)} bands")
    
    # Keep only valid bands
    datacube_clean = datacube[:, :, valid_bands]
    p_clean = len(valid_bands)
    
    print(f"  - New shape: ({m}, {n}, {p_clean})")
    
    # ============================================================
    # STEP 2: Flatten to (m*n, p_clean)
    # ============================================================
    print("\n[STEP 2] Flattening to (pixels, bands)...")
    
    X_flat = datacube_clean.reshape(-1, p_clean)
    print(f"  - Flattened shape: ({X_flat.shape[0]:,}, {X_flat.shape[1]})")
    
    # ============================================================
    # STEP 3: Remove sample vectors that are ALL NaN
    # ============================================================
    print("\n[STEP 3] Removing all-NaN sample vectors...")
    
    # Find samples (pixels) that are all NaN
    all_nan_samples = np.isnan(X_flat).all(axis=1)
    valid_sample_indices = ~all_nan_samples
    
    n_all_nan_samples = all_nan_samples.sum()
    n_valid_samples = valid_sample_indices.sum()
    
    print(f"  - All-NaN samples: {n_all_nan_samples:,}")
    print(f"  - Valid samples: {n_valid_samples:,}")
    
    # Extract only valid samples
    X_valid = X_flat[valid_sample_indices, :]
    print(f"  - Valid data shape: ({X_valid.shape[0]:,}, {X_valid.shape[1]})")
    
    # Remove any remaining NaNs (replace with 0 or mean)
    if np.isnan(X_valid).any():
        nan_count = np.isnan(X_valid).sum()
        print(f"  - Replacing {nan_count:,} remaining NaN values with zeros")
        X_valid = np.nan_to_num(X_valid, nan=0.0)
    
    # ============================================================
    # STEP 4: Run noise estimation on valid samples
    # ============================================================
    print("\n[STEP 4] Running noise estimation...")
    
    try:
        # Your noise estimation function
        w, Rw = est_noise(X_valid, noise_type='additive')
        print(f"  - Noise estimation successful!")
        print(f"  - Noise estimates shape: {w.shape}")
    except Exception as e:
        print(f"  - Error in noise estimation: {e}")
        print("  - Using fallback method...")
        # Fallback: use diagonal estimation
        w = np.std(X_valid, axis=0) * 0.1
        print(f"  - Fallback estimates shape: {w.shape}")
    
    # ============================================================
    # STEP 5: Reconstruct noise matrix with valid samples in original positions
    # ============================================================
    print("\n[STEP 5] Reconstructing noise cube...")
    
    # Create empty noise matrix with NaN for invalid pixels
    noise_matrix = np.full((total_pixels, p_clean), np.nan)
    
    # Place noise estimates at their original positions
    noise_matrix[valid_sample_indices, :] = w
    
    # Reshape back to (m, n, p_clean)
    noise_cube = noise_matrix.reshape(m, n, p_clean)
    print(f"  - Reconstructed noise cube shape: {noise_cube.shape}")
    print(f"  - Valid pixels with noise estimates: {np.isfinite(noise_cube).any(axis=2).sum():,}")
    print(f"  - Invalid pixels: {np.isnan(noise_cube).any(axis=2).sum():,}")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Original datacube: ({m}, {n}, {p})")
    print(f"Removed all-NaN bands: {len(all_nan_bands)}")
    print(f"Removed all-NaN pixels: {n_all_nan_samples:,}")
    print(f"Valid pixels: {n_valid_samples:,}")
    print(f"Final noise cube: ({m}, {n}, {p_clean})")
    print("=" * 60)
    
    return noise_cube, w, Rw, valid_sample_indices, datacube_clean









def save_noise_plots(noise_cube, output_dir='noise_plots', bands_to_save=None):
    """
    Save individual band plots to files.
    
    Parameters:
    -----------
    noise_cube : numpy.ndarray
        3D array of shape (m, n, p)
    output_dir : str
        Directory to save plots
    bands_to_save : list or None
        List of band indices to save. If None, saves all bands.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    m, n, p = noise_cube.shape
    
    if bands_to_save is None:
        bands_to_save = range(p)
    
    for band in bands_to_save:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(noise_cube[:, :, band], cmap='plasma', aspect='auto')
        ax.set_title(f'Noise - Band {band}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Save
        filename = os.path.join(output_dir, f'noise_band_{band:04d}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")







def plot_snr(snr, wavelengths=None, save_path=None):
    """
    Plot per-band SNR in dB.

    Parameters
    ----------
    snr_db      : np.ndarray (p,)
    wavelengths : np.ndarray (p,) or None
    save_path   : str or None
    """
    p = len(snr)
    x = wavelengths if wavelengths is not None else np.arange(p)
    xlabel = 'Wavelength (nm)' if wavelengths is not None else 'Band Index'

    best_idx  = np.nanargmax(snr)
    worst_idx = np.nanargmin(snr)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, snr, 'b-', linewidth=2.5, label='SNR (dB)')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('SNR', fontsize=12)
    ax.set_title('SNR vs Wavelength', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    plt.show()

    print(f"\nGlobal mean SNR : {np.nanmean(snr):.2f}")
    print(f"Best  band SNR  : {snr[best_idx]:.2f}  @ {x[best_idx]:.1f}")
    print(f"Worst band SNR  : {snr[worst_idx]:.2f} @ {x[worst_idx]:.1f}")







def compute_snr_from_hysime(datacube_clean, Rw, wavelengths=None):
    """
    Parameters
    ----------
    datacube_clean : (m, n, p)
    Rw             : (p, p) noise covariance from est_noise
    """
    m, n, p = datacube_clean.shape

    X = datacube_clean.reshape(-1, p)
    valid = ~np.isnan(X).all(axis=1)
    X_valid = np.nan_to_num(X[valid], nan=0.0)

    noise_std  = np.sqrt(np.maximum(np.diag(Rw), 1e-20))   # (p,)
    signal_std = np.std(X_valid, axis=0)                    # (p,)

    snr_lin = signal_std / noise_std
    snr_db  = 20 * np.log10(np.maximum(snr_lin, 1e-10))

    print(f"Signal std range: {signal_std.min():.4e} – {signal_std.max():.4e}")
    print(f"Noise  std range: {noise_std.min():.4e}  – {noise_std.max():.4e}")
    print(f"SNR range:        {snr_db.min():.1f} – {snr_db.max():.1f} dB")

    return snr_db, snr_lin












def main(l2a_nc_path):

    print(Panel(f"Running processing for {Path(l2a_nc_path).name}.", title="HYPSO Processing", expand=False))
    print(f"Processing started at {datetime.now()}")


    # Check if the first file exists
    if not os.path.isfile(l2a_nc_path):
        print(f"Error: The file '{l2a_nc_path}' does not exist.")
        return

    # Process the first file
    print(f"Processing file: {l2a_nc_path}")

    nc_file = Path(l2a_nc_path)

    satobj = Hypso(path=nc_file, verbose=True, label=RAD_CAL_COEFFS)





    datacube = satobj.l2a_cube["acolite_l2w"]

    datacube = datacube.to_numpy()




    plt.imshow(datacube[400:450,0:50,40])
    plt.savefig("test.png")



    datacube = datacube[400:450,0:50,:]

    noise_cube, w, Rw, valid_sample_indices, datacube_clean = process_datacube_for_noise_estimation(datacube)


    snr_db, snr_lin = compute_snr_from_hysime(datacube_clean, Rw, wavelengths=satobj.wavelengths)
    

    #save_noise_plots(noise_cube, output_dir='noise_plots')


    plot_snr(snr_lin, save_path="/home/camerop/noise_plots/snr.png")


    print(f"Processing has completed sucessfully for capture {l2a_nc_path}!")

            

    

   











if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        print("Usage: python process_l1d_dir.py <nc_dir_path>")
        

        l2a_nc_path = "/home/camerop/HYPSO_DATA_AOC/aeronetvenice_2025-05-14T10-45-06Z/aeronetvenice_2025-05-14T10-45-06Z-moved-l2a-acolite_l2w.nc"
        l2a_nc_path = Path(l2a_nc_path)


    else:
        dir_path = sys.argv[1]



    main(l2a_nc_path)

    gc.collect()
    sys.exit(0)

