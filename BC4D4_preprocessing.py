"""
BC4D4 Preprocessing Module

Implements the preprocessing pipeline from Jangir et al. (2025) paper:
"Harnessing the synergy of statistics and deep learning for BCI competition 4 dataset 4"

Key differences from FingerFlex preprocessing:
1. Uses Isolation Forest for outlier removal (not wavelets)
2. Works on raw ECoG signals (no spectrograms)
3. Transforms data to near-Gaussian distribution [-1, +1]
"""

import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional, List
import pathlib
import os


# =============================================================================
# Constants
# =============================================================================

FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']
SUBJECT_ELECTRODES = {
    1: 62,  # Subject 1: 62 electrodes
    2: 48,  # Subject 2: 48 electrodes
    3: 64,  # Subject 3: 64 electrodes
}


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_raw_data(data_path: str, subject: int = 1) -> Dict[str, np.ndarray]:
    """
    Load raw BCI Competition 4 Dataset 4 data.

    BC4D4 works on RAW signals, not wavelet spectrograms.

    Parameters
    ----------
    data_path : str
        Path to the data directory containing .mat files
    subject : int
        Subject number (1, 2, or 3)

    Returns
    -------
    dict
        Dictionary containing:
        - 'train_ecog': Training ECoG data (time, electrodes)
        - 'train_finger': Training finger flexion data (time, 5)
        - 'test_ecog': Test ECoG data (time, electrodes)
        - 'test_finger': Test finger flexion data (time, 5)
    """
    # Load training data
    train_file = os.path.join(data_path, f'sub{subject}_comp.mat')
    train_data = scipy.io.loadmat(train_file)

    # Load test labels
    test_file = os.path.join(data_path, f'sub{subject}_testlabels.mat')
    test_data = scipy.io.loadmat(test_file)

    return {
        'train_ecog': train_data['train_data'].astype('float64'),  # (time, electrodes)
        'train_finger': train_data['train_dg'].astype('float64'),  # (time, 5 fingers)
        'test_ecog': train_data['test_data'].astype('float64'),    # (time, electrodes)
        'test_finger': test_data['test_dg'].astype('float64'),     # (time, 5 fingers)
    }


# =============================================================================
# Statistical Analysis Functions
# =============================================================================

def compute_descriptive_stats(data: np.ndarray, finger_names: List[str] = None) -> pd.DataFrame:
    """
    Compute descriptive statistics for finger movement data.

    Replicates Table 2 from the BC4D4 paper showing:
    - Mean, Std, Min, 25%, 50%, 75%, Max for each finger

    Parameters
    ----------
    data : np.ndarray
        Finger movement data, shape (time, 5) or (5, time)
    finger_names : list
        Names for each finger column

    Returns
    -------
    pd.DataFrame
        Descriptive statistics table
    """
    if finger_names is None:
        finger_names = FINGER_NAMES

    # Ensure shape is (time, 5)
    if data.shape[0] == 5:
        data = data.T

    df = pd.DataFrame(data, columns=finger_names)
    stats = df.describe().T

    # Round to 2 decimal places like the paper
    stats = stats.round(2)

    return stats


def plot_box_plots(data: np.ndarray, title: str = "Finger Movement Distribution",
                   finger_names: List[str] = None, save_path: str = None):
    """
    Create box plots for finger movement data.

    Box plots are used in the paper to identify outliers and
    understand data distribution before/after Isolation Forest.

    Parameters
    ----------
    data : np.ndarray
        Finger movement data, shape (time, 5) or (5, time)
    title : str
        Plot title
    finger_names : list
        Names for each finger
    save_path : str
        Optional path to save the figure
    """
    if finger_names is None:
        finger_names = FINGER_NAMES

    # Ensure shape is (time, 5)
    if data.shape[0] == 5:
        data = data.T

    fig, ax = plt.subplots(figsize=(10, 6))

    df = pd.DataFrame(data, columns=finger_names)
    df.boxplot(ax=ax)

    ax.set_title(title)
    ax.set_ylabel('Finger Flexion Value')
    ax.set_xlabel('Finger')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Box plot saved to: {save_path}")

    return fig, ax


def plot_histograms(data: np.ndarray, title: str = "Finger Movement Histograms",
                    finger_names: List[str] = None, save_path: str = None):
    """
    Create histograms for finger movement data.

    Used to verify near-Gaussian distribution after Isolation Forest.

    Parameters
    ----------
    data : np.ndarray
        Finger movement data, shape (time, 5) or (5, time)
    title : str
        Plot title
    finger_names : list
        Names for each finger
    save_path : str
        Optional path to save the figure
    """
    if finger_names is None:
        finger_names = FINGER_NAMES

    # Ensure shape is (time, 5)
    if data.shape[0] == 5:
        data = data.T

    fig, axes = plt.subplots(1, 5, figsize=(15, 4))

    for i, (ax, name) in enumerate(zip(axes, finger_names)):
        ax.hist(data[:, i], bins=50, edgecolor='black', alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Histogram saved to: {save_path}")

    return fig, axes


# =============================================================================
# Isolation Forest Outlier Removal
# =============================================================================

def apply_isolation_forest(data: np.ndarray,
                          contamination: float = 'auto',
                          n_estimators: int = 100,
                          random_state: int = 42,
                          verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Isolation Forest to remove outliers from finger movement data.

    This is the KEY INNOVATION from the BC4D4 paper.

    Isolation Forest:
    - Randomly selects a feature
    - Randomly selects a split value between min and max
    - Recursively partitions data until isolation
    - Anomalies require fewer splits to isolate (shorter path length)

    After applying Isolation Forest, data should:
    - Have near-Gaussian distribution
    - Fall approximately in range [-1, +1]
    - Have minimal outliers

    Parameters
    ----------
    data : np.ndarray
        Finger movement data, shape (time, 5) or (5, time)
    contamination : float or 'auto'
        Expected proportion of outliers in the data
        'auto' lets sklearn estimate it
    n_estimators : int
        Number of trees in the forest
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Whether to print progress

    Returns
    -------
    tuple
        - cleaned_data: Data with outliers removed
        - outlier_mask: Boolean mask where True = inlier, False = outlier
    """
    # Ensure shape is (time, 5)
    if data.shape[0] == 5:
        data = data.T
        was_transposed = True
    else:
        was_transposed = False

    if verbose:
        print(f"Applying Isolation Forest...")
        print(f"  Input shape: {data.shape}")
        print(f"  Contamination: {contamination}")
        print(f"  N estimators: {n_estimators}")

    # Initialize Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )

    # Fit and predict
    # -1 = outlier, 1 = inlier
    predictions = iso_forest.fit_predict(data)

    # Create mask (True = keep, False = remove)
    inlier_mask = predictions == 1

    # Get cleaned data
    cleaned_data = data[inlier_mask]

    n_outliers = (~inlier_mask).sum()
    n_total = len(data)

    if verbose:
        print(f"  Outliers removed: {n_outliers} ({100*n_outliers/n_total:.2f}%)")
        print(f"  Remaining samples: {len(cleaned_data)}")

    # Transpose back if needed
    if was_transposed:
        cleaned_data = cleaned_data.T

    return cleaned_data, inlier_mask


def apply_isolation_forest_to_ecog(ecog_data: np.ndarray,
                                   finger_mask: np.ndarray,
                                   verbose: bool = True) -> np.ndarray:
    """
    Apply the same outlier mask from finger data to ECoG data.

    The paper applies Isolation Forest to finger movement data first,
    then removes corresponding time points from ECoG data to maintain alignment.

    Parameters
    ----------
    ecog_data : np.ndarray
        ECoG data, shape (time, electrodes)
    finger_mask : np.ndarray
        Boolean mask from apply_isolation_forest (True = keep)
    verbose : bool
        Whether to print progress

    Returns
    -------
    np.ndarray
        ECoG data with outlier time points removed
    """
    if verbose:
        print(f"Applying finger outlier mask to ECoG data...")
        print(f"  ECoG shape before: {ecog_data.shape}")

    cleaned_ecog = ecog_data[finger_mask]

    if verbose:
        print(f"  ECoG shape after: {cleaned_ecog.shape}")

    return cleaned_ecog


# =============================================================================
# Data Normalization (BC4D4 style)
# =============================================================================

def normalize_ecog_zscore(ecog_data: np.ndarray,
                          fit_data: np.ndarray = None,
                          verbose: bool = True) -> Tuple[np.ndarray, StandardScaler]:
    """
    Apply z-score normalization to ECoG data.

    BC4D4 uses simple z-score normalization, unlike FingerFlex which
    uses bandpass filtering, notch filtering, and CAR.

    Parameters
    ----------
    ecog_data : np.ndarray
        ECoG data to normalize, shape (time, electrodes)
    fit_data : np.ndarray
        Optional data to fit the scaler on (for consistent train/test normalization)
    verbose : bool
        Whether to print progress

    Returns
    -------
    tuple
        - normalized_data: Z-score normalized ECoG data
        - scaler: Fitted StandardScaler for later use
    """
    if verbose:
        print("Applying z-score normalization to ECoG data...")

    scaler = StandardScaler()

    if fit_data is not None:
        scaler.fit(fit_data)
    else:
        scaler.fit(ecog_data)

    normalized = scaler.transform(ecog_data)

    if verbose:
        print(f"  Mean: {normalized.mean():.6f}")
        print(f"  Std: {normalized.std():.6f}")

    return normalized, scaler


# =============================================================================
# Data Preparation for BC4D4 Model
# =============================================================================

def prepare_bc4d4_input(ecog_data: np.ndarray,
                        window_size: int = 1,
                        verbose: bool = True) -> np.ndarray:
    """
    Prepare ECoG data for BC4D4 model input.

    BC4D4 Input Format (from paper):
    - Shape: (batch, num_features, 1)
    - No wavelet spectrograms - raw ECoG signals only

    For window_size=1 (sample-by-sample prediction):
    - Input: (N_samples, N_electrodes, 1)

    For window_size>1 (windowed prediction):
    - Input: (N_windows, N_electrodes, window_size)

    Parameters
    ----------
    ecog_data : np.ndarray
        ECoG data, shape (time, electrodes)
    window_size : int
        Size of the input window (default=1 for sample-by-sample)
    verbose : bool
        Whether to print progress

    Returns
    -------
    np.ndarray
        Reshaped data for BC4D4 model input
    """
    if verbose:
        print(f"Preparing BC4D4 input format...")
        print(f"  Input shape: {ecog_data.shape}")
        print(f"  Window size: {window_size}")

    n_samples, n_electrodes = ecog_data.shape

    if window_size == 1:
        # Simple reshape: (time, electrodes) -> (time, electrodes, 1)
        bc4d4_input = ecog_data[:, :, np.newaxis]
    else:
        # Create sliding windows
        n_windows = n_samples - window_size + 1
        bc4d4_input = np.zeros((n_windows, n_electrodes, window_size))

        for i in range(n_windows):
            bc4d4_input[i] = ecog_data[i:i+window_size].T

    if verbose:
        print(f"  Output shape: {bc4d4_input.shape}")

    return bc4d4_input


def prepare_bc4d4_targets(finger_data: np.ndarray,
                          window_size: int = 1,
                          finger_idx: int = None,
                          verbose: bool = True) -> np.ndarray:
    """
    Prepare finger movement targets for BC4D4 model.

    BC4D4 trains SEPARATE models for EACH finger (output=1).
    This is different from FingerFlex which predicts all 5 fingers at once.

    Parameters
    ----------
    finger_data : np.ndarray
        Finger movement data, shape (time, 5) or (5, time)
    window_size : int
        Size of input window (to align with inputs)
    finger_idx : int
        Which finger to extract (0=thumb, 1=index, ..., 4=little)
        If None, returns all 5 fingers
    verbose : bool
        Whether to print progress

    Returns
    -------
    np.ndarray
        Target data for training
    """
    # Ensure shape is (time, 5)
    if finger_data.shape[0] == 5:
        finger_data = finger_data.T

    if verbose:
        print(f"Preparing BC4D4 targets...")
        print(f"  Input shape: {finger_data.shape}")

    n_samples = finger_data.shape[0]

    if window_size > 1:
        # Align targets with windowed inputs (use last value in window)
        n_windows = n_samples - window_size + 1
        targets = finger_data[window_size-1:]
    else:
        targets = finger_data

    if finger_idx is not None:
        targets = targets[:, finger_idx:finger_idx+1]
        if verbose:
            print(f"  Selected finger: {FINGER_NAMES[finger_idx]}")

    if verbose:
        print(f"  Output shape: {targets.shape}")

    return targets


# =============================================================================
# Complete BC4D4 Preprocessing Pipeline
# =============================================================================

def preprocess_bc4d4(data_path: str,
                     subject: int = 1,
                     contamination: float = 'auto',
                     window_size: int = 1,
                     save_plots: bool = False,
                     output_dir: str = None,
                     verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Complete BC4D4 preprocessing pipeline.

    Steps (following the paper):
    1. Load raw data
    2. Statistical analysis (before preprocessing)
    3. Apply Isolation Forest to finger data
    4. Apply same mask to ECoG data
    5. Z-score normalize ECoG
    6. Statistical analysis (after preprocessing)
    7. Prepare data in BC4D4 input format

    Parameters
    ----------
    data_path : str
        Path to data directory with .mat files
    subject : int
        Subject number (1, 2, or 3)
    contamination : float or 'auto'
        Isolation Forest contamination parameter
    window_size : int
        Input window size for the model
    save_plots : bool
        Whether to save statistical analysis plots
    output_dir : str
        Directory to save plots (required if save_plots=True)
    verbose : bool
        Whether to print progress

    Returns
    -------
    dict
        Dictionary containing preprocessed data:
        - 'train_ecog': Training ECoG (N, electrodes, window_size)
        - 'train_finger': Training finger data (N, 5)
        - 'test_ecog': Test ECoG (N, electrodes, window_size)
        - 'test_finger': Test finger data (N, 5)
        - 'stats_before': Statistics before preprocessing
        - 'stats_after': Statistics after preprocessing
    """
    if verbose:
        print("=" * 60)
        print(f"BC4D4 Preprocessing Pipeline - Subject {subject}")
        print("=" * 60)

    # Create output directory if needed
    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load raw data
    if verbose:
        print("\n[Step 1] Loading raw data...")
    raw_data = load_raw_data(data_path, subject)

    # Step 2: Statistical analysis BEFORE preprocessing
    if verbose:
        print("\n[Step 2] Statistical analysis (BEFORE Isolation Forest)...")
    stats_before = compute_descriptive_stats(raw_data['train_finger'])
    if verbose:
        print(stats_before)

    if save_plots and output_dir:
        plot_box_plots(raw_data['train_finger'],
                      title=f"Subject {subject} - Before Isolation Forest",
                      save_path=os.path.join(output_dir, f'sub{subject}_boxplot_before.png'))
        plot_histograms(raw_data['train_finger'],
                       title=f"Subject {subject} - Before Isolation Forest",
                       save_path=os.path.join(output_dir, f'sub{subject}_histogram_before.png'))
        plt.close('all')

    # Step 3: Apply Isolation Forest to finger data
    if verbose:
        print("\n[Step 3] Applying Isolation Forest to finger data...")
    train_finger_clean, train_mask = apply_isolation_forest(
        raw_data['train_finger'],
        contamination=contamination,
        verbose=verbose
    )

    test_finger_clean, test_mask = apply_isolation_forest(
        raw_data['test_finger'],
        contamination=contamination,
        verbose=verbose
    )

    # Step 4: Apply same mask to ECoG data
    if verbose:
        print("\n[Step 4] Applying outlier mask to ECoG data...")
    train_ecog_clean = apply_isolation_forest_to_ecog(
        raw_data['train_ecog'], train_mask, verbose=verbose
    )
    test_ecog_clean = apply_isolation_forest_to_ecog(
        raw_data['test_ecog'], test_mask, verbose=verbose
    )

    # Step 5: Z-score normalize ECoG
    if verbose:
        print("\n[Step 5] Z-score normalizing ECoG data...")
    train_ecog_norm, scaler = normalize_ecog_zscore(
        train_ecog_clean, verbose=verbose
    )
    test_ecog_norm, _ = normalize_ecog_zscore(
        test_ecog_clean, fit_data=train_ecog_clean, verbose=verbose
    )

    # Step 6: Statistical analysis AFTER preprocessing
    if verbose:
        print("\n[Step 6] Statistical analysis (AFTER Isolation Forest)...")
    stats_after = compute_descriptive_stats(train_finger_clean)
    if verbose:
        print(stats_after)

    if save_plots and output_dir:
        plot_box_plots(train_finger_clean,
                      title=f"Subject {subject} - After Isolation Forest",
                      save_path=os.path.join(output_dir, f'sub{subject}_boxplot_after.png'))
        plot_histograms(train_finger_clean,
                       title=f"Subject {subject} - After Isolation Forest",
                       save_path=os.path.join(output_dir, f'sub{subject}_histogram_after.png'))
        plt.close('all')

    # Step 7: Prepare BC4D4 input format
    if verbose:
        print("\n[Step 7] Preparing BC4D4 input format...")

    # Ensure finger data is (time, 5)
    if train_finger_clean.shape[0] == 5:
        train_finger_clean = train_finger_clean.T
    if test_finger_clean.shape[0] == 5:
        test_finger_clean = test_finger_clean.T

    train_ecog_bc4d4 = prepare_bc4d4_input(train_ecog_norm, window_size, verbose)
    test_ecog_bc4d4 = prepare_bc4d4_input(test_ecog_norm, window_size, verbose)

    train_finger_bc4d4 = prepare_bc4d4_targets(train_finger_clean, window_size, verbose=verbose)
    test_finger_bc4d4 = prepare_bc4d4_targets(test_finger_clean, window_size, verbose=verbose)

    if verbose:
        print("\n" + "=" * 60)
        print("Preprocessing complete!")
        print(f"  Train ECoG shape: {train_ecog_bc4d4.shape}")
        print(f"  Train Finger shape: {train_finger_bc4d4.shape}")
        print(f"  Test ECoG shape: {test_ecog_bc4d4.shape}")
        print(f"  Test Finger shape: {test_finger_bc4d4.shape}")
        print("=" * 60)

    return {
        'train_ecog': train_ecog_bc4d4,
        'train_finger': train_finger_bc4d4,
        'test_ecog': test_ecog_bc4d4,
        'test_finger': test_finger_bc4d4,
        'stats_before': stats_before,
        'stats_after': stats_after,
        'train_mask': train_mask,
        'test_mask': test_mask,
    }


def save_preprocessed_data(data: Dict[str, np.ndarray],
                           output_path: str,
                           subject: int = 1):
    """
    Save preprocessed BC4D4 data to disk.

    Parameters
    ----------
    data : dict
        Dictionary from preprocess_bc4d4()
    output_path : str
        Directory to save the data
    subject : int
        Subject number
    """
    os.makedirs(output_path, exist_ok=True)

    np.save(os.path.join(output_path, f'sub{subject}_train_ecog_bc4d4.npy'),
            data['train_ecog'])
    np.save(os.path.join(output_path, f'sub{subject}_train_finger_bc4d4.npy'),
            data['train_finger'])
    np.save(os.path.join(output_path, f'sub{subject}_test_ecog_bc4d4.npy'),
            data['test_ecog'])
    np.save(os.path.join(output_path, f'sub{subject}_test_finger_bc4d4.npy'),
            data['test_finger'])

    print(f"BC4D4 preprocessed data saved to: {output_path}")


def load_preprocessed_data(data_path: str, subject: int = 1) -> Dict[str, np.ndarray]:
    """
    Load preprocessed BC4D4 data from disk.

    Parameters
    ----------
    data_path : str
        Directory containing saved data
    subject : int
        Subject number

    Returns
    -------
    dict
        Dictionary with train/test ECoG and finger data
    """
    return {
        'train_ecog': np.load(os.path.join(data_path, f'sub{subject}_train_ecog_bc4d4.npy')),
        'train_finger': np.load(os.path.join(data_path, f'sub{subject}_train_finger_bc4d4.npy')),
        'test_ecog': np.load(os.path.join(data_path, f'sub{subject}_test_ecog_bc4d4.npy')),
        'test_finger': np.load(os.path.join(data_path, f'sub{subject}_test_finger_bc4d4.npy')),
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='BC4D4 Preprocessing Pipeline')
    parser.add_argument('--data_path', type=str, default='./data/pure_data',
                       help='Path to raw data directory')
    parser.add_argument('--output_path', type=str, default='./data/bc4d4',
                       help='Path to save preprocessed data')
    parser.add_argument('--subject', type=int, default=1, choices=[1, 2, 3],
                       help='Subject number')
    parser.add_argument('--contamination', type=float, default=None,
                       help='Isolation Forest contamination (default: auto)')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save statistical analysis plots')

    args = parser.parse_args()

    contamination = args.contamination if args.contamination else 'auto'

    # Run preprocessing
    data = preprocess_bc4d4(
        data_path=args.data_path,
        subject=args.subject,
        contamination=contamination,
        save_plots=args.save_plots,
        output_dir=args.output_path,
        verbose=True
    )

    # Save results
    save_preprocessed_data(data, args.output_path, args.subject)
