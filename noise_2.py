#!/usr/bin/env python3

import os
import sys
import numpy as np
from pathlib import Path
import gc
from rich import print
from rich.panel import Panel
from datetime import datetime
import matplotlib.pyplot as plt
import csv
import glob

import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

sys.path.insert(0, '/home/camerop/AC/hypso-package/hypso')
sys.path.insert(0, '/home/camerop/AC/hypso-package/hypso1_calibration')
sys.path.insert(0, '/home/camerop/AC/hypso-package/hypso2_calibration')

from hypso import Hypso

# Ensure the target directory exists
HYPSO_DATA_DIR = "/home/camerop/HYPSO_DATA_AOC/"



RAD_CAL_COEFFS = "moved"
OUTPUT_BASE_DIR = Path("/home/camerop/Output/")

SNR_CAPTURES_CSV = "/home/camerop/AC/hypso-ac-processing/config/snr_captures.csv"

SNR_BAND_START = 10   # skip noisy edge bands

if True:
    ATMOSPHERIC_CORRECTION_ALGS=["dps"]
    LABEL = "moved_unmasked"
else:
    ATMOSPHERIC_CORRECTION_ALGS=["polymer", "acolite_l2w", "l1d"]
    LABEL = "moved"



# =============================================================================
# NOISE ESTIMATION (Bioucas-Dias & Nascimento)
# =============================================================================

def est_noise(y, noise_type='additive'):
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

    y = y.T          # (p, N)
    L, N = y.shape

    if noise_type == 'poisson':
        sqy      = np.sqrt(y * (y > 0))
        u, _     = est_additive_noise(sqy)
        x        = (sqy - u) ** 2
        w        = np.sqrt(x) * u * 2
        Rw       = w @ w.T / N
    else:
        w, Rw = est_additive_noise(y)

    return w.T, Rw.T   # back to (N, p) and (p, p)


# =============================================================================
# PREPROCESSING
# =============================================================================

def prepare_and_estimate_noise(datacube, noise_type='additive'):
    """
    Prepare a (m, n, p) datacube and run HySime noise estimation.

    Steps
    -----
    1. Drop all-NaN bands
    2. Flatten to (m*n, p)
    3. Drop all-NaN pixels
    4. Replace any remaining NaNs with 0
    5. Run est_noise
    6. Reconstruct noise cube (m, n, p) with NaN for invalid pixels

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
    logging.info(f"\n{'='*60}")
    logging.info(f"NOISE ESTIMATION PREPROCESSING")
    logging.info(f"{'='*60}")
    logging.info(f"Input shape: ({m}, {n}, {p})")

    # --- Step 1: drop all-NaN bands ---
    valid_bands = [b for b in range(p) if not np.isnan(datacube[:, :, b]).all()]
    dropped     = p - len(valid_bands)
    logging.info(f"Dropped {dropped} all-NaN bands → {len(valid_bands)} remain")

    datacube_clean = datacube[:, :, valid_bands]
    p_clean = len(valid_bands)

    # --- Step 2: flatten ---
    X = datacube_clean.reshape(-1, p_clean)          # (m*n, p_clean)

    # --- Step 3: drop all-NaN pixels ---
    valid_mask = ~np.isnan(X).all(axis=1)
    X_valid    = X[valid_mask]
    logging.info(f"Dropped {(~valid_mask).sum():,} all-NaN pixels → {valid_mask.sum():,} remain")

    # --- Step 4: replace remaining NaNs ---
    if np.isnan(X_valid).any():
        n_remaining = np.isnan(X_valid).sum()
        logging.info(f"Replacing {n_remaining:,} remaining NaNs with 0")
        X_valid = np.nan_to_num(X_valid, nan=0.0)

    # --- Step 5: noise estimation ---
    logging.info("Running est_noise...")
    w, Rw = est_noise(X_valid, noise_type=noise_type)
    logging.info(f"Done. w: {w.shape}, Rw: {Rw.shape}")

    # --- Step 6: reconstruct noise cube ---
    noise_flat              = np.full((m * n, p_clean), np.nan)
    noise_flat[valid_mask]  = w
    noise_cube              = noise_flat.reshape(m, n, p_clean)

    logging.info(f"Output noise cube: {noise_cube.shape}")
    logging.info(f"{'='*60}\n")

    return datacube_clean, noise_cube, w, Rw, valid_mask, valid_bands


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

def plot_snr(snr_db, snr_lin, wavelengths=None, save_path=None):
    """
    Two-panel plot: SNR in dB (top) and linear (bottom).

    Parameters
    ----------
    snr_db      : np.ndarray (p,)
    snr_lin     : np.ndarray (p,)
    wavelengths : np.ndarray (p,) or None
    save_path   : str or None
    """
    x      = wavelengths if wavelengths is not None else np.arange(len(snr_db))
    xlabel = 'Wavelength (nm)' if wavelengths is not None else 'Band Index'

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(x, snr_db, 'b-', linewidth=2)
    axes[0].set_ylabel('SNR (dB)', fontsize=12)
    axes[0].set_title('SNR vs Wavelength', fontsize=14, fontweight='bold')
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

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()

    best  = np.nanargmax(snr_db)
    worst = np.nanargmin(snr_db)
    print(f"\nBest  band: {x[best]:.1f}  → {snr_db[best]:.1f} dB  ({snr_lin[best]:.1f})")
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













def plot_rgb_with_aoi_old(datacube, wavelengths, aoi_y_min, aoi_y_max, 
                      aoi_x_min, aoi_x_max, save_path=None,
                      title=None, figsize=(12, 10), 
                      rgb_bands=None, percentile_clip=(2, 98)):
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
    """
    from matplotlib.patches import Rectangle
    
    # Find RGB bands if not specified
    if rgb_bands is None:
        # Typical RGB wavelengths for remote sensing: Red ~640-670nm, Green ~550-560nm, Blue ~450-490nm
        # Find closest bands to these wavelengths
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
    
    # Percentile-based contrast stretching
    p_min, p_max = np.percentile(rgb[rgb > 0], percentile_clip)
    rgb_stretched = np.clip((rgb - p_min) / (p_max - p_min), 0, 1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot RGB
    ax.imshow(rgb_stretched, aspect='auto')
    
    # Draw AOI rectangle in RED
    rect = Rectangle((aoi_x_min, aoi_y_min), 
                     aoi_x_max - aoi_x_min, 
                     aoi_y_max - aoi_y_min,
                     linewidth=3, edgecolor='red', facecolor='none', 
                     alpha=0.9, linestyle='-')
    ax.add_patch(rect)
    
    # Add corner markers with larger size
    corners = [(aoi_x_min, aoi_y_min), (aoi_x_max, aoi_y_min),
               (aoi_x_min, aoi_y_max), (aoi_x_max, aoi_y_max)]
    for x, y in corners:
        ax.plot(x, y, 'r+', markersize=15, markeredgewidth=3)
    
    # Add coordinate labels at corners
    corner_labels = [
        (aoi_x_min, aoi_y_min, 'top-left'),
        (aoi_x_max, aoi_y_min, 'top-right'),
        (aoi_x_min, aoi_y_max, 'bottom-left'),
        (aoi_x_max, aoi_y_max, 'bottom-right')
    ]
    
    for x, y, pos in corner_labels:
        # Offset for label position to avoid overlapping with marker
        if 'top' in pos:
            y_offset = -15
            va = 'top'
        else:
            y_offset = 15
            va = 'bottom'
            
        if 'left' in pos:
            x_offset = 5
            ha = 'left'
        else:
            x_offset = -5
            ha = 'right'
            
        label = f'({x},{y})'
        ax.text(x + x_offset, y + y_offset, label, 
                color='white', fontsize=10, fontweight='bold',
                ha=ha, va=va,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8))
    
    # Add AOI label in the center
    center_x = (aoi_x_min + aoi_x_max) // 2
    center_y = (aoi_y_min + aoi_y_max) // 2
    ax.text(center_x, center_y, 'AOI', 
            color='red', fontsize=16, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
    
    # Add band wavelength info
    band_info = (f"R: {wavelengths[rgb_bands[0]]:.1f}nm, "
                 f"G: {wavelengths[rgb_bands[1]]:.1f}nm, "
                 f"B: {wavelengths[rgb_bands[2]]:.1f}nm")
    
    # Set title
    if title is None:
        title = f'RGB Composite with AOI\n{band_info}'
    else:
        title = f'{title}\n{band_info}'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Pixel X', fontsize=12)
    ax.set_ylabel('Pixel Y', fontsize=12)
    
    # Add grid with low opacity
    ax.grid(True, alpha=0.1, linestyle='--')
    
    # Add scale bar or dimension info
    height = aoi_y_max - aoi_y_min + 1
    width = aoi_x_max - aoi_x_min + 1
    ax.text(0.02, 0.98, f'AOI Size: {width} × {height} pixels', 
            transform=ax.transAxes, color='white', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
            verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"RGB plot saved to {save_path}")
    
    plt.show()
    plt.close()
    
    return rgb_stretched






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
# MAIN
# =============================================================================

def main(l2a_nc_path):

    logging.info(f"Opening list of HYPSO captures from {SNR_CAPTURES_CSV}")
    logging.info(f"HYPSO data directory is {HYPSO_DATA_DIR}")

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

        snr_capture_dict = snr_captures_dict[snr_capture_key]

        file = snr_capture_dict["path"]
        capture_name = snr_capture_dict["capture_name"]


        if not os.path.isfile(file):
            logging.warning(f"Error: '{file}' does not exist. Skipping.")
            continue

        satobj = Hypso(path=Path(file), verbose=True, label=LABEL)

        aca = snr_capture_dict["atmospheric_correction"].lower()

        try:
            if aca == "l1d":
                datacube = satobj.l1d_cube.to_numpy()
            else:
                datacube = satobj.l2a_cube[aca].to_numpy()
        except:
            logging.error(f"No product for {aca} found in {file}! Skipping file.")
            continue
            
        

        # Spatial subset for testing
        datacube_aoi = datacube[aoi_y_min:aoi_y_max, aoi_x_min:aoi_x_max, :]

        
        capture_dir = satobj.capture_dir
        snr_dir = Path(capture_dir, f"SNR_{aca}")
        os.makedirs(snr_dir, exist_ok=True)



        # Noise estimation
        
        datacube_clean, noise_cube, w, Rw, valid_mask, valid_bands = prepare_and_estimate_noise(datacube_aoi, noise_type='additive')

        wavelengths_used = satobj.wavelengths[valid_bands][SNR_BAND_START:]


        snr_db, snr_lin  = compute_snr_from_cubes(datacube_clean, noise_cube, band_start=SNR_BAND_START)


        # Plot


        plot_band = 40


        aoi_figure_filename = f"snr_aoi_band_{plot_band}_{capture_name}_{aca}.png"
        plt.imshow(datacube[:, :, plot_band])
        plt.savefig(Path(snr_dir, aoi_figure_filename))
        plt.close()

        aoi_figure_filename = f"snr_capture_rgb_{capture_name}_{aca}_rgb.png"
        plot_rgb_with_aoi(datacube, satobj.wavelengths, aoi_y_min, aoi_y_max,
                               aoi_x_min, aoi_x_max, save_path=Path(snr_dir, aoi_figure_filename))



        snr_figure_filename = f"snr_{capture_name}_{aca}.png"
        plot_snr(snr_db, snr_lin, wavelengths=wavelengths_used, save_path=Path(snr_dir, snr_figure_filename))

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