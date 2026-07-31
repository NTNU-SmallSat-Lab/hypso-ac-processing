#!/usr/bin/env python3

import os
import sys
import numpy as np
from pathlib import Path
import gc
from rich import print
from rich.panel import Panel
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import csv
import glob
import earthaccess
import pandas as pd
import xarray as xr
from scipy.signal import convolve2d
import json

from satpy import Scene
from pyresample.future.resamplers.nearest import KDTreeNearestXarrayResampler
from pyresample.bilinear.xarr import XArrayBilinearResampler 
from pyresample.geometry import SwathDefinition, AreaDefinition

import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

sys.path.insert(0, '/home/camerop/AC/hypso-package/hypso')
sys.path.insert(0, '/home/camerop/AC/hypso-package/hypso1_calibration')
sys.path.insert(0, '/home/camerop/AC/hypso-package/hypso2_calibration')

from hypso import Hypso


# Satellite Matchup Constants
# Short names for earthaccess lookup
SAT_LOOKUP = {
    "PACE": "PACE_OCI_L2_AOP_NRT",
    "AQUA": "MODISA_L2_OC",
    "TERRA": "MODIST_L2_OC",
    "NOAA-20": "VIIRSJ1_L2_OC",
    "NOAA-21": "VIIRSJ2_L2_OC",
    "SUOMI-NPP": "VIIRSN_L2_OC"
    }
    #"PACE": "PACE_OCI_L2_AOP_NRT",
    #"PACE": "PACE_OCI_L2_SFREFL",

# List l2 flags, then build them into a dict
l2_flags_list = [
    "ATMFAIL", "LAND", "PRODWARN", "HIGLINT", "HILT", "HISATZEN", "COASTZ",
    "SPARE", "STRAYLIGHT", "CLDICE", "COCCOLITH", "TURBIDW", "HISOLZEN",
    "SPARE", "LOWLW", "CHLFAIL", "NAVWARN", "ABSAER", "SPARE", "MAXAERITER",
    "MODGLINT", "CHLWARN", "ATMWARN", "SPARE", "SEAICE", "NAVFAIL", "FILTER",
    "SPARE", "BOWTIEDEL", "HIPOL", "PRODFAIL", "SPARE"]
L2_FLAGS = {flag: 1 << idx for idx, flag in enumerate(l2_flags_list)}

# Bailey and Werdell 2006 exclusion criteria
EXCLUSION_FLAGS = ["LAND", "HIGLINT", "HILT", "STRAYLIGHT", "CLDICE",
                   "ATMFAIL", "LOWLW", "FILTER", "NAVFAIL", "NAVWARN"]



# =============================================================================
# PROCESSING SETTINGS 
# =============================================================================


# Ensure the target directory exists
HYPSO_DATA_DIR = "/home/camerop/HYPSO_DATA_AOC/"

RAD_CAL_COEFFS = "moved"
OUTPUT_BASE_DIR = Path("/home/camerop/Output/")

SNR_CAPTURES_CSV = "/home/camerop/AC/hypso-ac-processing/config/snr_captures.csv"

SNR_BAND_START = 10   # skip noisy edge bands

if False:
    ATMOSPHERIC_CORRECTION_ALGS=["dps"]
    LABEL = "moved_unmasked"
else:
    ATMOSPHERIC_CORRECTION_ALGS=["polymer", "acolite_l2w", "l1d"]
    LABEL = "moved"

PACE_PRODUCTS = [("PACE_OCI_L2_AOP", "3.2"), ("PACE_OCI_L1C_SCI", "3"), ("PACE_OCI_L1B_SCI", "3")]

EARTHDATA_u = "cpenne"
EARTHDATA_p = "Dec1!onJG0@1LogoMen5un!"


# =============================================================================
# NOISE ESTIMATION (Bioucas-Dias & Nascimento)
# =============================================================================

def est_noise_bioucasdias2008(datacube, wavelengths, noise_type='additive'):
    """
    Infer noise in a hyperspectral dataset by assuming each band is well
    modelled by a linear regression on the remaining bands.

    Parameters
    ----------
    y : np.ndarray, shape (N, p)  — N pixels, p bands
    noise_type : 'additive' | 'poisson'

    Returns
    -------
    w  : np.ndarray (N, p) — per-pixel noise estimates
    Rw : np.ndarray (p, p) — diagonal noise covariance matrix
    """

    def est_additive_noise(r):
        # r is (L, N)
        small = 1e-6
        L, N = r.shape
        w = np.zeros((L, N), dtype=float)
        RR  = r @ r.T
        RRi = np.matrix(np.linalg.pinv(RR + small * np.eye(L)))
        for i in range(L):
            XX       = RRi - (RRi[:, i] * RRi[i, :]) / RRi[i, i]
            RRa      = RR[:, i].copy()
            RRa[i]   = 0
            beta     = np.dot(XX, RRa)
            beta[0, i] = 0
            w[i, :] = r[i, :] - np.dot(beta, r)
        Rw = np.diag(np.diag(w @ w.T / N))
        return w, Rw


    logging.info(f"Running Bioucas-Dias SNR analysis...")
    logging.info(f"Data shape: {datacube.shape}")

    M, valid_mask, valid_bands, valid_wavelengths = sanitize_datacube(datacube, wavelengths)




    m, n, p = datacube.shape
    p_clean = len(valid_bands)

    y = M

    logging.info(f"Running Bioucas-Dias noise estimation...")

    y = y.T  # (p, N)
    L, N = y.shape

    if noise_type == 'poisson':
        sqy      = np.sqrt(y * (y > 0))
        u, _     = est_additive_noise(sqy)
        x        = (sqy - u) ** 2
        w        = np.sqrt(x) * u * 2
        Rw       = w @ w.T / N
    else:
        w, Rw = est_additive_noise(y)

    w = w.T # back to (N, p)
    Rw = Rw.T # back to (p, p)
    logging.info(f"Done. w: {w.shape}, Rw: {Rw.shape}")
    

    # Reconstruct noise cube
    noise_flat              = np.full((m * n, p_clean), np.nan)
    noise_flat[valid_mask]  = w
    noise_cube              = noise_flat.reshape(m, n, p_clean)

    logging.info(f"Output noise cube: {noise_cube.shape}")


    # Trim to requested band range along the spectral axis (axis 2)
    M_signal = datacube[:, :, valid_bands]
    N_noise = noise_cube

    # Calculate noise standard deviation per band (ignoring NaNs spatially)
    # Axis (0, 1) collapses the 2D spatial dimensions to leave a per-band array
    noise_std = np.nanstd(N_noise, axis=(0, 1))
    noise_std = np.maximum(noise_std, 1e-20)  # Avoid division by zero

    # Calculate clean signal mean per band (ignoring NaNs spatially)
    signal_mean = np.nanmean(M_signal, axis=(0, 1))
    signal_mean = np.maximum(np.abs(signal_mean), 1e-20) # Protect log10 from zero/negatives

    # Calculate Linear and dB SNR values
    snr_lin = signal_mean / noise_std
    snr_db  = 20 * np.log10(np.maximum(snr_lin, 1e-10))

    print(f"--- SNR Analysis (Bands {valid_wavelengths[0]} to {valid_wavelengths[-1]}) ---")
    print(f"Mean SNR across bands: {np.nanmean(snr_lin):.2f}")
    print(f"Min SNR: {np.nanmin(snr_lin):.2f} (band {np.argmin(snr_lin)})")
    print(f"Max SNR: {np.nanmax(snr_lin):.2f} (band {np.argmax(snr_lin)})")

    print(f"Signal mean: {signal_mean.min():.4e} - {signal_mean.max():.4e}")
    print(f"Noise  std : {noise_std.min():.4e} - {noise_std.max():.4e}")
    print(f"SNR (dB)   : {snr_db.min():.1f} - {snr_db.max():.1f}")
    print(f"SNR (lin)  : {snr_lin.min():.1f} - {snr_lin.max():.1f}")


    statistics = {
        'noise': {
            'mean': np.nanmean(noise_std),
            'min': np.nanmin(noise_std),
            'max': np.nanmax(noise_std),
            'std': np.nanstd(noise_std),
            #'low_noise_bands': low_noise_bands,
            #'high_noise_bands': high_noise_bands,
            'array': noise_std
        },
        'signal': {
            #'mean': np.nanmean(signal_std),
            #'min': np.nanmin(signal_std),
            #'max': np.nanmax(signal_std),
            #'std': np.nanstd(signal_std),
            'mean_array': signal_mean,
            #'std_array': signal_std
        },
        'snr': {
            'mean_linear': np.nanmean(snr_lin),
            'min_linear': np.nanmin(snr_lin),
            'max_linear': np.nanmax(snr_lin),
            'min_band': int(np.argmin(snr_lin)),
            'max_band': int(np.argmax(snr_lin)),
            'mean_db': np.nanmean(snr_db),
            'min_db': np.nanmin(snr_db),
            'max_db': np.nanmax(snr_db),
            'linear': snr_lin,
            'db': snr_db
        },
        'wavelengths': valid_wavelengths,
        'signal_min': np.nanmin(signal_mean),
        'signal_max': np.nanmax(signal_mean),
        'noise_min': np.nanmin(noise_std),
        'noise_max': np.nanmax(noise_std)
    }


    #snr_wavelengths = valid_wavelengths

    #return snr_lin, snr_db, snr_wavelengths

    return statistics

# =============================================================================
# NOISE ESTIMATION (Green 1988)
# =============================================================================

# Code here is based on MATLAB code from https://github.com/NTNU-SmallSat-Lab/imagingpipeline/

def est_noise_green1988(datacube, wavelengths, axis=1):

    print(f"Running Green (1988) SNR analysis...")
    print(f"Data shape: {datacube.shape}")

    M, valid_mask, valid_bands, valid_wavelengths = sanitize_datacube(datacube, wavelengths)

    # --- Noise: difference adjacent pixels in the 3D cube, before flattening ---
    cube = datacube[:, :, valid_bands]          # same band subset as M

    if axis == 0:                                # along-track (successive frames)
        dX = cube[:-1, :, :] - cube[1:, :, :]
    else:                                        # across-track (detector spatial axis)
        dX = cube[:, :-1, :] - cube[:, 1:, :]

    dX = dX.reshape(-1, len(valid_bands))
    dX = dX[~np.isnan(dX).any(axis=1)]           # drop incomplete pairs, not pixels
    n_pairs = dX.shape[0]
    logging.info(f"Green: {n_pairs:,} valid adjacent pairs (axis={axis})")

    # Factor 0.5 because Var[x_k - x_k+1] = 2 * sigma^2
    noise_cov = 0.5 * (dX.T @ dX) / n_pairs
    noise_std = np.sqrt(np.diag(noise_cov))

    # --- Signal: flattened array from sanitize_datacube is fine here ---
    M_2d = M
    
    print("\nNoise Statistics:")
    print(f"Mean noise std across bands: {np.nanmean(noise_std):.4f}")
    print(f"Min noise std: {np.nanmin(noise_std):.4f}")
    print(f"Max noise std: {np.nanmax(noise_std):.4f}")
    print(f"Std of noise std: {np.nanstd(noise_std):.4f}")


    # Find bands with lowest/highest noise
    low_noise_bands = np.argsort(noise_std)[:5]
    high_noise_bands = np.argsort(noise_std)[-5:]
    
    print(f"\n5 bands with lowest noise: {low_noise_bands}")
    print(f"5 bands with highest noise: {high_noise_bands}")


    # Estimate signal as mean of each band
    signal_mean = np.nanmean(M_2d, axis=0)
    signal_std = np.nanstd(M_2d, axis=0)


    print("\Signal Statistics:")
    print(f"Mean signal std across bands: {np.nanmean(signal_std):.4f}")
    print(f"Min signal std: {np.nanmin(signal_std):.4f}")
    print(f"Max signal std: {np.nanmax(signal_std):.4f}")
    print(f"Std of signal std: {np.nanstd(signal_std):.4f}")


    # SNR calculations
    snr_mean = signal_mean / noise_std  # Using mean as signal
    snr_rms = signal_std / noise_std    # Using RMS as signal
    

    # SNR analysis
    #M_2d = M.reshape(-1, M.shape[2])
    signal_mean = np.nanmean(M_2d, axis=0)
    snr_lin = signal_mean / noise_std
    snr_db  = 20 * np.log10(np.maximum(snr_lin, 1e-10))

    print(f"--- SNR Analysis (Bands {valid_wavelengths[0]} to {valid_wavelengths[-1]}) ---")
    print(f"Mean SNR across bands: {np.nanmean(snr_lin):.2f}")
    print(f"Min SNR: {np.nanmin(snr_lin):.2f} (band {np.argmin(snr_lin)})")
    print(f"Max SNR: {np.nanmax(snr_lin):.2f} (band {np.argmax(snr_lin)})")

    print(f"Signal mean: {signal_mean.min():.4e} - {signal_mean.max():.4e}")
    print(f"Noise  std : {noise_std.min():.4e} - {noise_std.max():.4e}")
    print(f"SNR (dB)   : {snr_db.min():.1f} - {snr_db.max():.1f}")
    print(f"SNR (lin)  : {snr_lin.min():.1f} - {snr_lin.max():.1f}")

    statistics = {
        'noise': {
            'mean': np.nanmean(noise_std),
            'min': np.nanmin(noise_std),
            'max': np.nanmax(noise_std),
            'std': np.nanstd(noise_std),
            'low_noise_bands': low_noise_bands,
            'high_noise_bands': high_noise_bands,
            'array': noise_std
        },
        'signal': {
            'mean': np.nanmean(signal_std),
            'min': np.nanmin(signal_std),
            'max': np.nanmax(signal_std),
            'std': np.nanstd(signal_std),
            'mean_array': signal_mean,
            'std_array': signal_std
        },
        'snr': {
            'mean_linear': np.nanmean(snr_lin),
            'min_linear': np.nanmin(snr_lin),
            'max_linear': np.nanmax(snr_lin),
            'min_band': int(np.argmin(snr_lin)),
            'max_band': int(np.argmax(snr_lin)),
            'mean_db': np.nanmean(snr_db),
            'min_db': np.nanmin(snr_db),
            'max_db': np.nanmax(snr_db),
            'linear': snr_lin,
            'db': snr_db
        },
        'wavelengths': valid_wavelengths,
        'signal_min': np.nanmin(signal_mean),
        'signal_max': np.nanmax(signal_mean),
        'noise_min': np.nanmin(noise_std),
        'noise_max': np.nanmax(noise_std)
    }

    #snr_wavelengths = valid_wavelengths

    #return snr_lin, snr_db, snr_wavelengths, noise_cov, noise_std

    return statistics



def est_noise_green1988_old(datacube, wavelengths):

    print(f"Running Green (1988) SNR analysis...")
    print(f"Data shape: {datacube.shape}")

    M, valid_mask, valid_bands, valid_wavelengths = sanitize_datacube(datacube, wavelengths)

    # Convert to 2D if needed
    if M.ndim == 3:
        h, w, d = M.shape
        M_2d = M.reshape(h * w, d)
    else:
        M_2d = M
        d = M_2d.shape[1]
    
    m, n = M_2d.shape
    
    # Compute spatial differences (noise estimate)
    dX = np.zeros((m-1, n))
    for i in range(m-1):
        dX[i, :] = M_2d[i, :] - M_2d[i+1, :]
    
    # Estimate noise covariance
    # The factor 0.5 is because variance of difference = 2 * variance of noise
    noise_cov = 0.5 * (dX.T @ dX) / (m - 1)
    
    # Get noise standard deviation per band
    noise_std = np.sqrt(np.diag(noise_cov))
    
    print("\nNoise Statistics:")
    print(f"Mean noise std across bands: {np.nanmean(noise_std):.4f}")
    print(f"Min noise std: {np.nanmin(noise_std):.4f}")
    print(f"Max noise std: {np.nanmax(noise_std):.4f}")
    print(f"Std of noise std: {np.nanstd(noise_std):.4f}")


    # Find bands with lowest/highest noise
    low_noise_bands = np.argsort(noise_std)[:5]
    high_noise_bands = np.argsort(noise_std)[-5:]
    
    print(f"\n5 bands with lowest noise: {low_noise_bands}")
    print(f"5 bands with highest noise: {high_noise_bands}")


    # Estimate signal as mean of each band
    signal_mean = np.nanmean(M_2d, axis=0)
    signal_std = np.nanstd(M_2d, axis=0)


    print("\Signal Statistics:")
    print(f"Mean signal std across bands: {np.nanmean(signal_std):.4f}")
    print(f"Min signal std: {np.nanmin(signal_std):.4f}")
    print(f"Max signal std: {np.nanmax(signal_std):.4f}")
    print(f"Std of signal std: {np.nanstd(signal_std):.4f}")


    # SNR calculations
    snr_mean = signal_mean / noise_std  # Using mean as signal
    snr_rms = signal_std / noise_std    # Using RMS as signal
    

    # SNR analysis
    #M_2d = M.reshape(-1, M.shape[2])
    signal_mean = np.nanmean(M_2d, axis=0)
    snr_lin = signal_mean / noise_std
    snr_db  = 20 * np.log10(np.maximum(snr_lin, 1e-10))

    print(f"--- SNR Analysis (Bands {valid_wavelengths[0]} to {valid_wavelengths[-1]}) ---")
    print(f"Mean SNR across bands: {np.nanmean(snr_lin):.2f}")
    print(f"Min SNR: {np.nanmin(snr_lin):.2f} (band {np.argmin(snr_lin)})")
    print(f"Max SNR: {np.nanmax(snr_lin):.2f} (band {np.argmax(snr_lin)})")

    print(f"Signal mean: {signal_mean.min():.4e} - {signal_mean.max():.4e}")
    print(f"Noise  std : {noise_std.min():.4e} - {noise_std.max():.4e}")
    print(f"SNR (dB)   : {snr_db.min():.1f} - {snr_db.max():.1f}")
    print(f"SNR (lin)  : {snr_lin.min():.1f} - {snr_lin.max():.1f}")

    statistics = {
        'noise': {
            'mean': np.nanmean(noise_std),
            'min': np.nanmin(noise_std),
            'max': np.nanmax(noise_std),
            'std': np.nanstd(noise_std),
            'low_noise_bands': low_noise_bands,
            'high_noise_bands': high_noise_bands,
            'array': noise_std
        },
        'signal': {
            'mean': np.nanmean(signal_std),
            'min': np.nanmin(signal_std),
            'max': np.nanmax(signal_std),
            'std': np.nanstd(signal_std),
            'mean_array': signal_mean,
            'std_array': signal_std
        },
        'snr': {
            'mean_linear': np.nanmean(snr_lin),
            'min_linear': np.nanmin(snr_lin),
            'max_linear': np.nanmax(snr_lin),
            'min_band': int(np.argmin(snr_lin)),
            'max_band': int(np.argmax(snr_lin)),
            'mean_db': np.nanmean(snr_db),
            'min_db': np.nanmin(snr_db),
            'max_db': np.nanmax(snr_db),
            'linear': snr_lin,
            'db': snr_db
        },
        'wavelengths': valid_wavelengths,
        'signal_min': np.nanmin(signal_mean),
        'signal_max': np.nanmax(signal_mean),
        'noise_min': np.nanmin(noise_std),
        'noise_max': np.nanmax(noise_std)
    }

    #snr_wavelengths = valid_wavelengths

    #return snr_lin, snr_db, snr_wavelengths, noise_cov, noise_std

    return statistics


# =============================================================================
# PREPROCESSING
# =============================================================================

def sanitize_datacube(datacube, wavelengths, noise_type='additive'):
    """
    Prepare a (m, n, p) datacube and run HySime noise estimation.

    Steps
    -----
    1. Drop all-NaN bands
    2. Flatten to (m*n, p)
    3. Drop all-NaN pixels
    4. Replace any remaining NaNs with 0
    5. Reconstruct noise cube (m, n, p) with NaN for invalid pixels

    Parameters
    ----------
    datacube   : np.ndarray (m, n, p)
    noise_type : 'additive' | 'poisson'

    Returns
    -------
    datacube_clean : np.ndarray (m, n, p_clean)  — NaN bands removed
    noise_cube     : np.ndarray (m, n, p_clean)  — per-pixel noise, NaN where invalid
    w              : np.ndarray (valid_N, p_clean) — raw noise estimates
    Rw             : np.ndarray (p_clean, p_clean) — diagonal noise covariance
    valid_mask     : np.ndarray (m*n,) bool        — which pixels were used
    valid_bands    : list[int]                      — which original bands were kept
    """
    m, n, p = datacube.shape

    logging.info(f"Sanitizing and reshaping datacube")
    logging.info(f"Input shape: ({m}, {n}, {p})")

    # --- Step 1: drop all-NaN bands ---
    valid_bands = [b for b in range(p) if not np.isnan(datacube[:, :, b]).all()]
    dropped     = p - len(valid_bands)
    logging.info(f"Dropped {dropped} all-NaN bands → {len(valid_bands)} remain")

    datacube_clean = datacube[:, :, valid_bands]
    p_clean = len(valid_bands)

    # --- Step 2: flatten ---
    M = datacube_clean.reshape(-1, p_clean) # (m*n, p_clean)

    # --- Step 3: drop all-NaN pixels ---
    valid_mask = ~np.isnan(M).all(axis=1)
    M_valid    = M[valid_mask]
    logging.info(f"Dropped {(~valid_mask).sum():,} all-NaN pixels → {valid_mask.sum():,} remain")

    # --- Step 4: replace remaining NaNs ---
    if np.isnan(M_valid).any():
        n_remaining = np.isnan(M_valid).sum()
        logging.info(f"Replacing {n_remaining:,} remaining NaNs with 0")
        M_valid = np.nan_to_num(M_valid, nan=0.0)


    valid_wavelengths = wavelengths[valid_bands]

    return M_valid, valid_mask, valid_bands, valid_wavelengths


# =============================================================================
# SNR
# =============================================================================

def compute_snr(datacube_clean, Rw, band_start=0):
    """
    Compute per-band classic SNR (mean / std) using HySime noise covariance.

    SNR = signal_mean / noise_std  (linear contrast-based SNR)
    SNR_dB = 20 * log10(SNR)

    Parameters
    ----------
    datacube_clean : np.ndarray (m, n, p)
    Rw             : np.ndarray (p, p) diagonal noise covariance matrix from HySime
    band_start     : int — skip the first N bands (e.g. 10 to drop noisy edge bands)

    Returns
    -------
    snr_db  : np.ndarray (p_used,)
    snr_lin : np.ndarray (p_used,)
    """
    # Flatten and drop invalid pixels
    m, n, p = datacube_clean.shape
    X       = datacube_clean.reshape(-1, p)
    
    # Keep rows where NOT all elements are NaN
    valid   = ~np.isnan(X).all(axis=1)
    X_valid = np.nan_to_num(X[valid], nan=0.0)

    # Trim to requested band range
    X_valid = X_valid[:, band_start:]
    Rw_used = Rw[band_start:, band_start:]

    # Extract noise standard deviation from HySime covariance diagonal
    noise_std   = np.sqrt(np.maximum(np.diag(Rw_used), 1e-20))
    
    # Calculate the classic mean signal (with absolute value to protect log10)
    signal_mean = np.maximum(np.abs(np.mean(X_valid, axis=0)), 1e-20)

    # Linear contrast-based SNR computation
    snr_lin = signal_mean / noise_std
    snr_db  = 20 * np.log10(np.maximum(snr_lin, 1e-10))

    # Updated print logs with correct statistical terminology
    print(f"Signal mean: {signal_mean.min():.4e} – {signal_mean.max():.4e}")
    print(f"Noise  std : {noise_std.min():.4e}  – {noise_std.max():.4e}")
    print(f"SNR (dB)   : {snr_db.min():.1f} – {snr_db.max():.1f}")
    print(f"SNR (lin)  : {snr_lin.min():.1f} – {snr_lin.max():.1f}")

    return snr_db, snr_lin


def compute_snr_from_cubes(datacube_clean, noise_cube, band_start=0):
    """
    Compute per-band classic SNR (mean / std) directly using the clean HSI data
    and the estimated noise cube.

    SNR = signal_mean / noise_std
    SNR_dB = 20 * log10(SNR)

    Parameters
    ----------
    datacube_clean : np.ndarray (m, n, p) - Clean isolated datacube
    noise_cube     : np.ndarray (m, n, p) - Residual noise cube from estimation
    band_start     : int — skip the first N bands (e.g. 10 to drop noisy edge bands)

    Returns
    -------
    snr_db  : np.ndarray (p_used,)
    snr_lin : np.ndarray (p_used,)
    """
    # Trim to requested band range along the spectral axis (axis 2)
    X_clean = datacube_clean[:, :, band_start:]
    N_noise = noise_cube[:, :, band_start:]

    # Calculate noise standard deviation per band (ignoring NaNs spatially)
    # Axis (0, 1) collapses the 2D spatial dimensions to leave a per-band array
    noise_std = np.nanstd(N_noise, axis=(0, 1))
    noise_std = np.maximum(noise_std, 1e-20)  # Avoid division by zero

    # Calculate clean signal mean per band (ignoring NaNs spatially)
    signal_mean = np.nanmean(X_clean, axis=(0, 1))
    signal_mean = np.maximum(np.abs(signal_mean), 1e-20) # Protect log10 from zero/negatives

    # Calculate Linear and dB SNR values
    snr_lin = signal_mean / noise_std
    snr_db  = 20 * np.log10(np.maximum(snr_lin, 1e-10))

    # Print logs using correct statistical terminology
    print(f"--- SNR Analysis (Bands {band_start} to {datacube_clean.shape[2]}) ---")
    print(f"Signal mean: {signal_mean.min():.4e} – {signal_mean.max():.4e}")
    print(f"Noise  std : {noise_std.min():.4e}  – {noise_std.max():.4e}")
    print(f"SNR (dB)   : {snr_db.min():.1f} – {snr_db.max():.1f}")
    print(f"SNR (lin)  : {snr_lin.min():.1f} – {snr_lin.max():.1f}")

    return snr_db, snr_lin



# =============================================================================
# PLOTTING
# =============================================================================

def plot_snr(snr_db, snr_lin, wl_min=400, wl_max=800, atmospheric_correction=None, wavelengths=None, save_path=None):
    """
    Two-panel plot: SNR in dB (top) and linear (bottom).

    Parameters
    ----------
    snr_db      : np.ndarray (p,)
    snr_lin     : np.ndarray (p,)
    atmospheric_correction: str or None
    wavelengths : np.ndarray (p,) or None
    save_path   : str or None
    """

    if atmospheric_correction is None:
        aca = ""
    else:
        aca = atmospheric_correction

    x      = wavelengths if wavelengths is not None else np.arange(len(snr_db))
    xlabel = 'Wavelength (nm)' if wavelengths is not None else 'Band Index'

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(x, snr_db, 'b-', linewidth=2,)
    axes[0].set_ylabel('SNR (dB)', fontsize=12)
    axes[0].set_title(f'SNR vs Wavelength {aca}', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(np.nanmean(snr_db), color='r', linestyle='--',
                    alpha=0.6, label=f'Mean: {np.nanmean(snr_db):.1f} dB')
    axes[0].legend()

    axes[1].plot(x, snr_lin, 'g-', linewidth=2)
    axes[1].set_ylabel('SNR (linear)', fontsize=12)
    axes[1].set_xlabel(xlabel, fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(np.nanmean(snr_lin), color='r', linestyle='--',
                    alpha=0.6, label=f'Mean: {np.nanmean(snr_lin):.1f}')
    axes[1].legend()


    axes[0].set_xlim(wl_min, wl_max)
    axes[1].set_xlim(wl_min, wl_max)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()

    best  = np.nanargmax(snr_db)
    worst = np.nanargmin(snr_db)
    print(f"\nBest band: {x[best]:.1f}  → {snr_db[best]:.1f} dB  ({snr_lin[best]:.1f})")
    print(f"Worst band: {x[worst]:.1f} → {snr_db[worst]:.1f} dB ({snr_lin[worst]:.1f})")
    print(f"Mean SNR:   {np.nanmean(snr_db):.1f} dB  ({np.nanmean(snr_lin):.1f} linear)")

    plt.close()


def print_aoi_minimal(aoi_y_min, aoi_y_max, aoi_x_min, aoi_x_max):
    """
    Minimal visualization - just the corner markers.
    """
    width = aoi_x_max - aoi_x_min + 1
    height = aoi_y_max - aoi_y_min + 1
    
    print(f"\nAOI: [{aoi_x_min},{aoi_x_max}] × [{aoi_y_min},{aoi_y_max}]")
    print()
    
    # Create a tiny representation
    print(f"  Y={aoi_y_min}  X={aoi_x_min} ┌{'─' * 20}┐ X={aoi_x_max}")
    print(f"                │{' ' * 20}│")
    print(f"                │{' ' * 20}│  Y={aoi_y_min + (height//2)}")
    print(f"                │{' ' * 20}│")
    print(f"  Y={aoi_y_max}  X={aoi_x_min} └{'─' * 20}┘ X={aoi_x_max}")
    
    print(f"\n  {width} columns × {height} rows")


def plot_rgb_with_aoi(datacube, wavelengths, aoi_y_min, aoi_y_max, 
                      aoi_x_min, aoi_x_max, save_path=None,
                      title=None, figsize=(12, 10), 
                      rgb_bands=None, percentile_clip=(2, 98),
                      bin_x=3, bin_y=1):
    """
    Plot RGB composite of the datacube with AOI box overlay in red.
    
    Parameters
    ----------
    datacube : np.ndarray (m, n, p) - The full datacube
    wavelengths : np.ndarray (p,) - Wavelengths for each band
    aoi_y_min, aoi_y_max : int - Y bounds of AOI
    aoi_x_min, aoi_x_max : int - X bounds of AOI
    save_path : str or Path - Where to save the figure
    title : str - Title for the plot
    figsize : tuple - Figure size
    rgb_bands : list - [red_band_idx, green_band_idx, blue_band_idx] 
                If None, uses approximate true color bands
    percentile_clip : tuple - (min_percentile, max_percentile) for contrast stretching
    bin_x : int - Binning factor in x-direction (default: 3)
    bin_y : int - Binning factor in y-direction (default: 1)
    """
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Find RGB bands if not specified
    if rgb_bands is None:
        # Typical RGB wavelengths for remote sensing: Red ~640-670nm, Green ~550-560nm, Blue ~450-490nm
        target_rgb = {
            'red': 660,    # nm
            'green': 560,  
            'blue': 480
        }
        
        rgb_bands = []
        for color, target_wl in target_rgb.items():
            # Find closest wavelength
            idx = np.argmin(np.abs(wavelengths - target_wl))
            rgb_bands.append(idx)
            logging.info(f"{color.capitalize()} band selected: {wavelengths[idx]:.1f} nm (index {idx})")
    
    # Extract RGB bands
    red_band = datacube[:, :, rgb_bands[0]]
    green_band = datacube[:, :, rgb_bands[1]]
    blue_band = datacube[:, :, rgb_bands[2]]
    
    # Stack into RGB array
    rgb = np.stack([red_band, green_band, blue_band], axis=-1)
    
    # Handle NaN and negative values
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Store original shape for coordinate adjustment
    original_shape = rgb.shape
    original_width = original_shape[1]
    original_height = original_shape[0]
    
    # Apply binning (averaging) in x and y directions
    if bin_x > 1 or bin_y > 1:
        # Calculate new dimensions (floor division to handle remainders)
        new_height = original_height // bin_y
        new_width = original_width // bin_x
        
        # Trim to exact multiples
        height_trim = new_height * bin_y
        width_trim = new_width * bin_x
        
        # Trim the array
        rgb_trimmed = rgb[:height_trim, :width_trim, :]
        
        # Reshape for binning
        # Shape: (new_height, bin_y, new_width, bin_x, 3)
        rgb_reshaped = rgb_trimmed.reshape(new_height, bin_y, new_width, bin_x, 3)
        
        # Average over bin_y and bin_x dimensions
        rgb_binned = np.mean(rgb_reshaped, axis=(1, 3))
        
        # Replace rgb with binned version
        rgb = rgb_binned
        
        # Adjust AOI coordinates for binning
        aoi_x_min_binned = aoi_x_min // bin_x
        aoi_x_max_binned = (aoi_x_max + 1) // bin_x - 1  # Adjust for inclusive indexing
        aoi_y_min_binned = aoi_y_min // bin_y
        aoi_y_max_binned = (aoi_y_max + 1) // bin_y - 1
        
        # Ensure coordinates are within bounds
        aoi_x_min_binned = max(0, min(aoi_x_min_binned, new_width - 1))
        aoi_x_max_binned = max(0, min(aoi_x_max_binned, new_width - 1))
        aoi_y_min_binned = max(0, min(aoi_y_min_binned, new_height - 1))
        aoi_y_max_binned = max(0, min(aoi_y_max_binned, new_height - 1))
        
        logging.info(f"Binned RGB shape: {rgb.shape}")
        logging.info(f"Binned AOI X: {aoi_x_min_binned}->{aoi_x_max_binned}, Y: {aoi_y_min_binned}->{aoi_y_max_binned}")
        
    else:
        aoi_x_min_binned = aoi_x_min
        aoi_x_max_binned = aoi_x_max
        aoi_y_min_binned = aoi_y_min
        aoi_y_max_binned = aoi_y_max
        new_height = original_height
        new_width = original_width
    
    # Percentile-based contrast stretching
    # Use only positive values for percentile calculation
    positive_mask = rgb > 0
    if np.any(positive_mask):
        p_min, p_max = np.percentile(rgb[positive_mask], percentile_clip)
    else:
        p_min, p_max = 0, np.max(rgb)
    
    # Avoid division by zero
    if p_max - p_min < 1e-10:
        p_max = p_min + 1e-10
    
    rgb_stretched = np.clip((rgb - p_min) / (p_max - p_min), 0, 1)
    
    # Create figure with appropriate size for binned image
    if bin_x > 1:
        # Adjust figure size to maintain aspect ratio
        aspect_ratio = new_width / new_height
        figsize_adjusted = (figsize[0], figsize[0] / aspect_ratio)
        fig, ax = plt.subplots(figsize=figsize_adjusted)
    else:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Plot RGB
    ax.imshow(rgb_stretched, aspect='auto', interpolation='nearest')
    
    # Draw AOI rectangle in RED (using binned coordinates)
    rect = Rectangle((aoi_x_min_binned, aoi_y_min_binned), 
                     aoi_x_max_binned - aoi_x_min_binned, 
                     aoi_y_max_binned - aoi_y_min_binned,
                     linewidth=3, edgecolor='red', facecolor='none', 
                     alpha=0.9, linestyle='-')
    ax.add_patch(rect)
    
    # Add corner markers with larger size
    corners = [(aoi_x_min_binned, aoi_y_min_binned), 
               (aoi_x_max_binned, aoi_y_min_binned),
               (aoi_x_min_binned, aoi_y_max_binned), 
               (aoi_x_max_binned, aoi_y_max_binned)]
    for x, y in corners:
        ax.plot(x, y, 'r+', markersize=15, markeredgewidth=3)
    
    # Add coordinate labels at corners (show original coordinates)
    corner_labels = [
        (aoi_x_min_binned, aoi_y_min_binned, 'top-left', aoi_x_min, aoi_y_min),
        (aoi_x_max_binned, aoi_y_min_binned, 'top-right', aoi_x_max, aoi_y_min),
        (aoi_x_min_binned, aoi_y_max_binned, 'bottom-left', aoi_x_min, aoi_y_max),
        (aoi_x_max_binned, aoi_y_max_binned, 'bottom-right', aoi_x_max, aoi_y_max)
    ]
    
    for x, y, pos, orig_x, orig_y in corner_labels:
        # Offset for label position to avoid overlapping with marker
        if 'top' in pos:
            y_offset = -max(10, new_height * 0.02)
            va = 'top'
        else:
            y_offset = max(10, new_height * 0.02)
            va = 'bottom'
            
        if 'left' in pos:
            x_offset = max(5, new_width * 0.01)
            ha = 'left'
        else:
            x_offset = -max(5, new_width * 0.01)
            ha = 'right'
            
        # Show both binned and original coordinates
        if bin_x > 1 or bin_y > 1:
            label = f'({orig_x},{orig_y})'
        else:
            label = f'({x},{y})'
            
        ax.text(x + x_offset, y + y_offset, label, 
                color='white', fontsize=10, fontweight='bold',
                ha=ha, va=va,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8))
    
    # Add AOI label in the center (using binned coordinates)
    center_x = (aoi_x_min_binned + aoi_x_max_binned) // 2
    center_y = (aoi_y_min_binned + aoi_y_max_binned) // 2
    ax.text(center_x, center_y, 'AOI', 
            color='red', fontsize=16, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
    
    # Add band wavelength info
    band_info = (f"R: {wavelengths[rgb_bands[0]]:.1f}nm, "
                 f"G: {wavelengths[rgb_bands[1]]:.1f}nm, "
                 f"B: {wavelengths[rgb_bands[2]]:.1f}nm")
    
    # Add binning info
    if bin_x > 1 or bin_y > 1:
        bin_info = f" | Binned: {bin_x}× (X), {bin_y}× (Y) | Shape: {new_width}×{new_height}"
    else:
        bin_info = ""
    
    # Set title
    if title is None:
        title = f'RGB Composite with AOI{bin_info}\n{band_info}'
    else:
        title = f'{title}{bin_info}\n{band_info}'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Pixel X' + (' (binned)' if (bin_x > 1 or bin_y > 1) else ''), fontsize=12)
    ax.set_ylabel('Pixel Y' + (' (binned)' if (bin_x > 1 or bin_y > 1) else ''), fontsize=12)
    
    # Add grid with low opacity
    ax.grid(True, alpha=0.1, linestyle='--')
    
    # Add scale bar or dimension info
    height = aoi_y_max - aoi_y_min + 1
    width = aoi_x_max - aoi_x_min + 1
    binned_height = aoi_y_max_binned - aoi_y_min_binned + 1
    binned_width = aoi_x_max_binned - aoi_x_min_binned + 1
    
    if bin_x > 1 or bin_y > 1:
        size_info = (f'AOI Size: {width}×{height} pixels (original) → '
                     f'{binned_width}×{binned_height} pixels (binned)')
    else:
        size_info = f'AOI Size: {width} × {height} pixels'
    
    # Add image dimensions info
    dim_info = f'Image: {original_width}×{original_height}'
    if bin_x > 1 or bin_y > 1:
        dim_info += f' → {new_width}×{new_height}'
    
    info_text = f"{size_info}\n{dim_info}"
    
    ax.text(0.02, 0.98, info_text, 
            transform=ax.transAxes, color='white', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
            verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"RGB plot saved to {save_path}")
        logging.info(f"  Original shape: {original_height}×{original_width}")
        if bin_x > 1 or bin_y > 1:
            logging.info(f"  Binned shape: {new_height}×{new_width}")
    
    plt.show()
    plt.close()
    
    return rgb_stretched





# =============================================================================
# PACE
# =============================================================================

def grab_pace_granules(start_date, 
                        end_date, 
                        latitude, 
                        longitude, 
                        sat="PACE",
                        selected_dates = None,
                        hypso_datetime = None, 
                        local_path = None, 
                        pace_product = ("PACE_OCI_L2_AOP", "3.2")):
    """
    Download and process satellite data for matchups.

    Caution: If the date or coordinates aren't formatted correctly, it might
    pull a huge granule list and take forever to run. If it takes more than 45
    seconds to print the number of granules, just kill the process.

    Uses the earthaccess package. Defaults to the PACE OCI L2 IOP datasets,
    but other satellites can be used if they have a corresponding short_name
    in the SAT_LOOKUP dictionary.

    Workflow:
        1. Get list of matchup granules

    Parameters
    ----------
    start_date : datetime or str
        Beginning of Aeronet data to run.
    end_date : datetime or str
        End of Aeronet data to run.
    latitude : float
        In decimal degrees for Aeronet-OC site for matchups
    longitude : float
        In decimal degrees (negative West) for Aeronet-OC site for matchups
    sat : str
        Name of satellite to search. Must be in SAT_LOOKUP dict constant.
    selected_dates : list of str, optional
        If given, only pull granules if the dates are in this list

    Returns
    -------
    pandas DataFrame object
        Flattened table of all satellite granule matchups.

    """


    def read_pace_data(file, latitude, longitude, rrs_wavelengths, granule_date):

        with xr.open_dataset(file, group="navigation_data") as ds_nav:
            sat_lat = ds_nav['latitude'].values
            sat_lon = ds_nav['longitude'].values

            y_dim = ds_nav.dims["number_of_lines"]
            x_dim = ds_nav.dims["pixels_per_line"]

        # Extract the data
        with xr.open_dataset(file, group="geophysical_data") as ds_data:
            rrs_data = ds_data['Rrs'].values
            flags_data = ds_data['l2_flags'].values

        # Calculate the bitwise OR of all flags in EXCLUSION_FLAGS to get a mask
        exclude_mask = sum(L2_FLAGS[flag] for flag in EXCLUSION_FLAGS)

        # Create a boolean mask
        # True means the flag value does not contain any of the EXCLUSION_FLAGS
        valid_mask = np.bitwise_and(flags_data, exclude_mask) == 0

        # Get stats and averages
        #if valid_mask.any():
        #    rrs_data = np.where(valid_mask[..., None], rrs_data, np.nan)

        #sat_lat = sat_lat.reshape((y_dim, x_dim))
        #sat_lon = sat_lon.reshape((y_dim, x_dim))
        #rrs_data = rrs_data.reshape((y_dim, x_dim, -1))




        return rrs_data, sat_lat, sat_lon


    # Look up short name from constants
    if sat not in SAT_LOOKUP.keys():
        raise ValueError(f"{sat} is not in the lookup dictionary. Available "
                         f"sats are: {', '.join(SAT_LOOKUP)}")
    short_name = SAT_LOOKUP[sat]

    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    # Format search parameters
    time_bounds = (f"{start_date}T00:00:00", f"{end_date}T23:59:59")

    # Run Earthaccess data search
    #results_old = earthaccess.search_data(point=(longitude, latitude),
    #                                  temporal=time_bounds,
    #                                  short_name=short_name)
    # NB: short_name was renamed from "PACE_OCI_L2_AOP_NRT" to "PACE_OCI_L2_AOP"




    short_name = pace_product[0]
    version = pace_product[1]

    results = earthaccess.search_data(point=(longitude, latitude),
                                    temporal=time_bounds,
                                    short_name=short_name,
                                    version=version)
    logging.info("Earthaccess granule search results:")
    print(results)

    
    if selected_dates is not None:
        filtered_results = [
            result for result in results
            if result["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"][:10]
            in selected_dates
            ]
        logging.info(f"Filtered to {len(filtered_results)} Granules.")
        #files = earthaccess.open(filtered_results)
        selected_results = filtered_results
    else:
        #files = earthaccess.open(results)
        selected_results = results

    if not results:
        print("No granules found")
        return None
    
    closest_granule = None
    closest_diff = None

    # Add UTC timezone if naive
    if hypso_datetime.tzinfo is None:
        hypso_datetime = hypso_datetime.replace(tzinfo=timezone.utc)
        logging.info(f"Made hypso_datetime aware (UTC): {hypso_datetime}")    

    for g in results:
        # Get the time string
        time_str = g['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
        logging.info(f"Processing granule: {g['umm']['GranuleUR']}")
        logging.info(f"Time string: {time_str}")
        
        # Parse to datetime
        granule_time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        if granule_time.tzinfo is None:
            granule_time = granule_time.replace(tzinfo=timezone.utc)
        logging.info(f"Parsed time: {granule_time}")
        
        # Calculate difference
        diff = abs(granule_time - hypso_datetime)
        logging.info(f"Difference from target: {diff}")
        
        # Check if this is the closest so far
        if closest_diff is None or diff < closest_diff:
            closest_diff = diff
            closest_granule = g
            logging.info(f"*** NEW CLOSEST!***")


    selected_results = [closest_granule]


    logging.info("Selected granules:")
    print(selected_results)

    open_remote_file = False

    try:
        logging.info(f"Downloading {str(short_name)} granule files to {local_path}")
        files = earthaccess.download(results, local_path=local_path, show_progress=True)
        logging.info("Downloaded granules:")
        print(files)


        try:
            logging.info("Checking if files can be opened")
            for file in files:
                with xr.open_dataset(file):
                    logging.info(f"Succeeded at opening {file}!")
                    pass
        except Exception as ex:
            logging.info(ex)
            logging.info("Corrupt file detected! Attempting to re-download with force=True argument.")
            logging.info(f"Downloading {str(short_name)} granule files to {local_path}")
            files = earthaccess.download(results, local_path=local_path, show_progress=True, force=True) 
            logging.info("Downloaded granules:")
            logging.info(files)


    except Exception:
        open_remote_file = True

    if open_remote_file:
        logging.info(f"Opening {str(short_name)} granule files from S3/HTTPS")
        files = earthaccess.open(selected_results)
        logging.info("Opened granules:")
        print(files)

    if len(files) == 0:
        logging.info("No granules found!")
        return None

    if short_name == "PACE_OCI_L2_AOP":

        # Pull out Rrs wavelengths for easier processing
        with xr.open_dataset(files[0], group="sensor_band_parameters") as ds_bands:
            rrs_wavelengths = ds_bands["wavelength_3d"].values

        # Loop through files and process
        sat_rrs_rows = []
        for idx, file in enumerate(files):
            
            try:
                granule_date = pd.to_datetime(file.granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"])
            except:
                ds = xr.open_dataset(file)
                granule_date = pd.to_datetime(ds.attrs['time_coverage_start']) 
                granule_date = granule_date.floor('s') 
            

            logging.info(f"Running Granule: {granule_date}")
            pace_rrs, pace_latitudes, pace_longitudes = read_pace_data(file, latitude, longitude, rrs_wavelengths, granule_date)

            return pace_rrs, rrs_wavelengths, pace_latitudes, pace_longitudes



    '''
    if short_name == "PACE_OCI_L1C_SCI":

        # Pull out Rrs wavelengths for easier processing
        try:
            with xr.open_dataset(files[0], group="sensor_views_bands") as ds_bands:
                Lwn_wavelengths = ds_bands["intensity_wavelength"].values[0] # Two views in L1C
        except Exception as ex:
            print(f"NetCDF file {files[0]} is likely corrupt. Unable to load data.")
            break

        # Loop through files and process
        sat_Lwn_rows = []
        for idx, file in enumerate(files):
            
            try:
                granule_date = pd.to_datetime(file.granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"])
            except:
                ds = xr.open_dataset(file)
                granule_date = pd.to_datetime(ds.attrs['time_coverage_start']) 
                granule_date = granule_date.floor('s') 
            

            print(f"Running Granule: {granule_date}")
            #row = get_fivebyfive_Lwn(file, latitude, longitude, Lwn_wavelengths, granule_date)
            #sat_Lwn_rows.append(row)
    '''

    '''
    if short_name == "PACE_OCI_L1B_SCI":

        # Pull out Rrs wavelengths for easier processing
        try:
            with xr.open_dataset(files[0], group="sensor_band_parameters") as ds_bands:
                rhot_blue_wavelengths = ds_bands["blue_wavelength"].values
                rhot_red_wavelengths = ds_bands["red_wavelength"].values
                rhot_wavelengths = np.concatenate([rhot_blue_wavelengths, rhot_red_wavelengths])
        except Exception as ex:
            logging.info(ex)
            logging.info(f"NetCDF file {files[0]} is likely corrupt. Unable to load data.")
            break

        # Loop through files and process
        sat_rhot_rows = []
        for idx, file in enumerate(files):
            
            try:
                granule_date = pd.to_datetime(file.granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"])
            except:
                ds = xr.open_dataset(file)
                granule_date = pd.to_datetime(ds.attrs['time_coverage_start']) 
                granule_date = granule_date.floor('s') 
            

            logging.info(f"Running Granule: {granule_date}")
            #row = get_fivebyfive_rhot(file, latitude, longitude, rhot_wavelengths, granule_date)
            #sat_rhot_rows.append(row)
    '''

    return


def resample_pace(satobj, pace_data, pace_latitudes, pace_longitudes):

    hypso_latitudes = satobj.latitudes
    hypso_longitudes = satobj.longitudes

    pace_longitudes = xr.DataArray(pace_longitudes, dims=['y', 'x'])
    pace_latitudes = xr.DataArray(pace_latitudes, dims=['y', 'x'])
    
    pace_swath_def = SwathDefinition(lons=pace_longitudes, lats=pace_latitudes)

    hypso_latitudes = xr.DataArray(hypso_latitudes, dims=['y', 'x'])
    hypso_longitudes = xr.DataArray(hypso_longitudes, dims=['y', 'x'])

    hypso_swath_def = SwathDefinition(lons=hypso_longitudes, lats=hypso_latitudes)

    pace_data = xr.DataArray(pace_data, dims=['y', 'x', 'bands'])

    nnrs = KDTreeNearestXarrayResampler(source_geo_def=pace_swath_def, target_geo_def=hypso_swath_def)
    #pace_data = nnrs.resample(pace_data, fill_value=np.nan, radius_of_influence=500)

    pace_data = nnrs.resample(pace_data, fill_value=np.nan)

    pace_data = pace_data.to_numpy()

    return pace_data







def find_darkest_aoi(datacube, window_size=50, valid=1):
    """
    Fast vectorized version returning exclusive bounds.
    """
    mean_spectral = np.nanmean(datacube, axis=2)
    
    valid_mask = ~np.isnan(mean_spectral)
    data_clean = np.nan_to_num(mean_spectral, nan=0)
    kernel = np.ones((window_size, window_size))
    
    sums = convolve2d(data_clean, kernel, mode='valid')
    valid_counts = convolve2d(valid_mask.astype(float), kernel, mode='valid')
    
    valid_counts[valid_counts == 0] = 1
    means = sums / valid_counts
    
    min_valid = window_size * window_size * valid
    means[valid_counts < min_valid] = np.inf
    
    min_idx = np.unravel_index(np.argmin(means), means.shape)
    best_y, best_x = min_idx
    
    # Return exclusive bounds
    return best_y, best_y + window_size, best_x, best_x + window_size





# =============================================================================
# UTILS
# =============================================================================


# Convert numpy arrays to lists for JSON serialization
def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj

# Save dictionary to JSON file
def save_dict_json(data_dict, filename='statistics.json'):
    # Convert numpy arrays to lists
    json_compatible = {k: convert_numpy(v) for k, v in data_dict.items()}
    
    with open(filename, 'w') as f:
        json.dump(json_compatible, f, indent=4, default=convert_numpy)
    print(f"Saved to {filename}")










# =============================================================================
# MAIN
# =============================================================================

def main(l2a_nc_path):

    logging.info(f"Opening list of HYPSO captures from {SNR_CAPTURES_CSV}")
    logging.info(f"HYPSO data directory is {HYPSO_DATA_DIR}")


    try:
        auth = earthaccess.login(persist=True)
        earthaccess_login = True
        logging.info("NASA Earthaccess login successful!")
    except earthaccess.LoginAttemptFailure:
        logging.warning("NASA Earthaccess login failed!")
        earthaccess_login = False 

    snr_captures_dict = {}

    # Read the CSV and download each file
    with open(SNR_CAPTURES_CSV, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:  # Skip empty rows
                #print(row)


                capture_name = row[0].strip()

                logging.info(f"Running search(es) for {capture_name} HYPSO capture")

                aoi_y_min = int(row[1].strip())
                aoi_y_max = int(row[2].strip())
                aoi_x_min = int(row[3].strip())
                aoi_x_max = int(row[4].strip())

                dir_path = Path(HYPSO_DATA_DIR, capture_name)

                for aca in ATMOSPHERIC_CORRECTION_ALGS:
                    
                    logging.info(f"Running search for {aca} atmospheric correction")

                    if os.path.isdir(dir_path):
                        # Pattern for the file: dirname + "-moved-l2a-polymer.nc"
                        # The dirname format example: "aeronetvenice_2025-05-14T10-45-06Z"
                        if aca == "l1d":
                            product_level = aca
                            pattern = os.path.join(dir_path, f"{capture_name}-{LABEL}-{product_level}.nc")
                        else:
                            pattern = os.path.join(dir_path, f"{capture_name}-{LABEL}-l2a-{aca}.nc")
                        
                        logging.info(f"Searching for NetCDF file using pattern {pattern}")

                        # Search for matching files
                        matching_files_list = glob.glob(pattern)

                        if len(matching_files_list) > 0:

                            if not os.path.isfile(matching_files_list[0]):
                                logging.error(f"The file '{matching_files_list[0]}' does not exist.")
                                continue
                            
                            file = matching_files_list[0]

                            logging.info(f"Matched file found: {file}")

                            print_aoi_minimal(aoi_y_min, aoi_y_max, aoi_x_min, aoi_x_max)

                            aoi_dict = {

                                "aoi_y_min": aoi_y_min,
                                "aoi_y_max": aoi_y_max,
                                "aoi_x_min": aoi_x_min,
                                "aoi_x_max": aoi_x_max,

                            }

                            snr_capture_dict = {

                                "path": file,
                                "aoi": aoi_dict,
                                "atmospheric_correction": aca,
                                "capture_name": capture_name,
                                "label": LABEL

                            }

                            snr_captures_dict[file] = snr_capture_dict

                        else:
                            logging.error(f"No matching files for {capture_name}!")
                    else:
                        continue

    print(snr_captures_dict)
    

    print(Panel(f"Running signal-to-noise analysis processing",
                title="SNR Processing", expand=False))
    logging.info(f"Processing started at {datetime.now()}")



    for snr_capture_key in snr_captures_dict.keys():

        plotting_dict = {}

        snr_capture_dict = snr_captures_dict[snr_capture_key]

        file = snr_capture_dict["path"]
        capture_name = snr_capture_dict["capture_name"]


        if not os.path.isfile(file):
            logging.warning(f"Error: '{file}' does not exist. Skipping.")
            continue

        satobj = Hypso(path=Path(file), verbose=True, label=LABEL)

        wl_range = satobj.wavelengths[SNR_BAND_START:]

        if True and earthaccess_login:

            capture_datetime = satobj.capture_datetime

            center_row = satobj.latitudes.shape[0] // 2
            center_col = satobj.latitudes.shape[1] // 2
            capture_target_lat = satobj.latitudes[center_row, center_col]

            center_row = satobj.longitudes.shape[0] // 2
            center_col = satobj.longitudes.shape[1] // 2
            capture_target_lon = satobj.longitudes[center_row, center_col]

            logging.info(f"HYPSO center: ({capture_target_lat}, {capture_target_lon})")

            capture_dir = Path(satobj.capture_dir)

            pace_rrs, \
            pace_wavelengths, \
            pace_latitudes, \
            pace_longitudes = grab_pace_granules(start_date = capture_datetime, 
                                                    end_date = capture_datetime, 
                                                    latitude = capture_target_lat, 
                                                    longitude = capture_target_lon, 
                                                    sat="PACE",
                                                    local_path=capture_dir,
                                                    hypso_datetime=capture_datetime,
                                                    pace_product = ("PACE_OCI_L2_AOP", "3.2") )
            
            pace_datacube = resample_pace(satobj, pace_rrs, pace_latitudes, pace_longitudes)
            pace_aoi_y_min, pace_aoi_y_max, pace_aoi_x_min, pace_aoi_x_max = find_darkest_aoi(datacube=pace_datacube, window_size=100, valid=1)
            pace_datacube_aoi = pace_datacube[pace_aoi_y_min:pace_aoi_y_max, pace_aoi_x_min:pace_aoi_x_max, :]



            # Calculate mean and noise
            #mean_signal = np.nanmean(pace_datacube_aoi, axis=(0, 1))
            #noise = np.nanstd(pace_datacube_aoi, axis=(0, 1))
            #snr = np.divide(mean_signal, noise, out=np.full_like(mean_signal, np.nan), where=noise!=0)
            #print(snr)


            plotting_dict['PACE'] = {
                "datacube": pace_datacube,
                "datacube_aoi": pace_datacube_aoi,
                "wavelengths": pace_wavelengths,
                "aca": "L2_AOP",
                "aoi_y_min": pace_aoi_y_min, 
                "aoi_y_max": pace_aoi_y_max, 
                "aoi_x_min": pace_aoi_x_min,
                "aoi_x_max": pace_aoi_x_max
            }

        else:
            logging.info("Skipping PACE download")
        



        

        try:

            aca = snr_capture_dict["atmospheric_correction"].lower()

            aoi_y_min = snr_capture_dict["aoi"]["aoi_y_min"]
            aoi_y_max = snr_capture_dict["aoi"]["aoi_y_max"]
            aoi_x_min = snr_capture_dict["aoi"]["aoi_x_min"]
            aoi_x_max = snr_capture_dict["aoi"]["aoi_x_max"]
            

            if aca == "l1d":
                datacube = satobj.l1d_cube.to_numpy()[:,:,SNR_BAND_START:]
                wavelengths = satobj.wavelengths[SNR_BAND_START:]
            else:
                datacube = satobj.l2a_cube[aca].to_numpy()[:,:,SNR_BAND_START:]
                wavelengths = satobj.wavelengths[SNR_BAND_START:]

            # Spatial subset 
            datacube_aoi = datacube[aoi_y_min:aoi_y_max, aoi_x_min:aoi_x_max, :]

            plotting_dict['HYPSO'] = {
                "datacube": datacube,
                "datacube_aoi": datacube_aoi,
                "wavelengths": wavelengths,
                "aca": aca,
                "aoi_y_min": aoi_y_min, 
                "aoi_y_max": aoi_y_max, 
                "aoi_x_min": aoi_x_min,
                "aoi_x_max": aoi_x_max
            }

        except Exception as ex:
            logging.error(f"No product for {aca} found in {file}! Skipping file.")
            print(ex)
            continue
            
        


        

        for key in plotting_dict.keys():

            sensor = key

            logging.info(f"Computing noise and SNR for {sensor}")
            #print(plotting_dict[key])

            plotting_dict[key]
            datacube = plotting_dict[key]["datacube"]
            datacube_aoi = plotting_dict[key]["datacube_aoi"]
            wavelengths = plotting_dict[key]["wavelengths"]
            aca = plotting_dict[key]["aca"]
            aoi_y_min = plotting_dict[key]["aoi_y_min"]
            aoi_y_max = plotting_dict[key]["aoi_y_max"]
            aoi_x_min = plotting_dict[key]["aoi_x_min"]
            aoi_x_max = plotting_dict[key]["aoi_x_max"]


            capture_dir = satobj.capture_dir
            snr_dir = Path(capture_dir, f"SNR_{aca}")
            os.makedirs(snr_dir, exist_ok=True)

            #datacube_aoi = np.clip(datacube_aoi, 0, 1)

            # snr_lin_green1988, snr_db_green1988, snr_wavelengths, _, _= est_noise_green1988(datacube_aoi, wavelengths)
            # snr_lin_bioucasdias2008, snr_db_bioucasdias2008, snr_wavelengths = est_noise_bioucasdias2008(datacube_aoi, wavelengths)

            statistics_green1988 = est_noise_green1988(datacube_aoi, wavelengths)
            statistics_bioucasdias2008 = est_noise_bioucasdias2008(datacube_aoi, wavelengths)

            snr_wavelengths = statistics_green1988['wavelengths']
            snr_wavelengths = statistics_bioucasdias2008['wavelengths']

            snr_lin_green1988 = statistics_green1988['snr']['linear']
            snr_lin_bioucasdias2008 = statistics_bioucasdias2008['snr']['linear']

            snr_db_green1988 = statistics_green1988['snr']['db']
            snr_db_bioucasdias2008 = statistics_bioucasdias2008['snr']['db']


            # Plotting
            plot_band = 40

            aoi_figure_filename = f"snr_{sensor}_aoi_band_{plot_band}_{capture_name}_{aca}.png"
            plt.imshow(datacube[:, :, plot_band])
            plt.savefig(Path(snr_dir, aoi_figure_filename))
            plt.close()

            aoi_figure_filename = f"snr_{sensor}_capture_rgb_{capture_name}_{aca}_rgb.png"
            plot_rgb_with_aoi(datacube, wavelengths, aoi_y_min, aoi_y_max,
                                aoi_x_min, aoi_x_max, save_path=Path(snr_dir, aoi_figure_filename))



            snr_green1988_figure_filename = f"snr_{sensor}_green1988_{capture_name}_{aca}.png"
            plot_snr(snr_db_green1988, snr_lin_green1988, wavelengths=snr_wavelengths, atmospheric_correction=aca, save_path=Path(snr_dir, snr_green1988_figure_filename))

            snr_bioucasdias2008_figure_filename = f"snr_{sensor}_bioucasdias2008_{capture_name}_{aca}.png"
            plot_snr(snr_db_bioucasdias2008, snr_lin_bioucasdias2008, wavelengths=snr_wavelengths, atmospheric_correction=aca, save_path=Path(snr_dir, snr_bioucasdias2008_figure_filename))

            snr_green1988_text_filename = Path(snr_dir, f"snr_{sensor}_green1988_{capture_name}_{aca}_stats.txt")
            snr_bioucasdias2008_text_filename = Path(snr_dir, f"snr_{sensor}_bioucasdias2008_{capture_name}_{aca}_stats.txt")
            

            save_dict_json(statistics_green1988, snr_green1988_text_filename)
            save_dict_json(statistics_bioucasdias2008, snr_bioucasdias2008_text_filename)

        logging.info(f"SNR processing done: {file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        l2a_nc_path = Path("/home/camerop/HYPSO_DATA_AOC/aeronetvenice_2025-05-14T10-45-06Z/"
                           "aeronetvenice_2025-05-14T10-45-06Z-moved-l2a-acolite_l2w.nc")
        
        l2a_nc_path = Path("/home/camerop/HYPSO_DATA_AOC/aeronetvenice_2025-05-14T10-45-06Z/"
                           "aeronetvenice_2025-05-14T10-45-06Z-moved-l1d.nc")
        

        l2a_nc_path = Path("/home/camerop/HYPSO_DATA_AOC/frohavet_2025-02-25T11-26-39Z/frohavet_2025-02-25T11-26-39Z-moved-l2a-acolite_l2w.nc")

    else:
        l2a_nc_path = Path(sys.argv[1])

    main(l2a_nc_path)
    gc.collect()
    sys.exit(0)