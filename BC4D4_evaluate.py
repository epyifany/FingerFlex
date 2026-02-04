"""
BC4D4 Evaluation Script

Evaluates BC4D4 models and compares results with paper benchmarks.

Expected results from Jangir et al. (2025) paper:

With Softsign Activation (BEST):
| Subject | Thumb | Index | Middle | Ring  | Little | Avg   |
|---------|-------|-------|--------|-------|--------|-------|
| Sub-1   | 0.88  | 0.86  | 0.84   | 0.86  | 0.86   | 0.86  |
| Sub-2   | 0.82  | 0.81  | 0.79   | 0.83  | 0.80   | 0.81  |
| Sub-3   | 0.91  | 0.85  | 0.91   | 0.93  | 0.90   | 0.90  |
| Overall |       |       |        |       |        | 0.85  |

With Tanh Activation:
| Subject | Thumb | Index | Middle | Ring  | Little | Avg   |
|---------|-------|-------|--------|-------|--------|-------|
| Sub-1   | 0.89  | 0.87  | 0.85   | 0.86  | 0.88   | 0.87  |
| Sub-2   | 0.79  | 0.66  | 0.45   | 0.70  | 0.78   | 0.68  |
| Sub-3   | 0.89  | 0.86  | 0.90   | 0.93  | 0.83   | 0.88  |
| Overall |       |       |        |       |        | 0.81  |

FingerFlex Baseline:
| Subject | Avg   |
|---------|-------|
| Sub-1   | 0.66  |
| Sub-2   | 0.62  |
| Sub-3   | 0.74  |
| Overall | 0.67  |
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
import pandas as pd

from BC4D4_model import BC4D4, create_bc4d4_model
from BC4D4_preprocessing import FINGER_NAMES, SUBJECT_ELECTRODES


# =============================================================================
# Paper Benchmarks
# =============================================================================

# Expected results from BC4D4 paper (Softsign activation)
PAPER_RESULTS_SOFTSIGN = {
    1: {'Thumb': 0.88, 'Index': 0.86, 'Middle': 0.84, 'Ring': 0.86, 'Little': 0.86, 'Average': 0.86},
    2: {'Thumb': 0.82, 'Index': 0.81, 'Middle': 0.79, 'Ring': 0.83, 'Little': 0.80, 'Average': 0.81},
    3: {'Thumb': 0.91, 'Index': 0.85, 'Middle': 0.91, 'Ring': 0.93, 'Little': 0.90, 'Average': 0.90},
}

# Expected results from BC4D4 paper (Tanh activation)
PAPER_RESULTS_TANH = {
    1: {'Thumb': 0.89, 'Index': 0.87, 'Middle': 0.85, 'Ring': 0.86, 'Little': 0.88, 'Average': 0.87},
    2: {'Thumb': 0.79, 'Index': 0.66, 'Middle': 0.45, 'Ring': 0.70, 'Little': 0.78, 'Average': 0.68},
    3: {'Thumb': 0.89, 'Index': 0.86, 'Middle': 0.90, 'Ring': 0.93, 'Little': 0.83, 'Average': 0.88},
}

# FingerFlex baseline (from paper comparison)
FINGERFLEX_RESULTS = {
    1: {'Average': 0.66},
    2: {'Average': 0.62},
    3: {'Average': 0.74},
}

OVERALL_BENCHMARKS = {
    'FingerFlex': 0.67,
    'BC4D4_Tanh': 0.81,
    'BC4D4_Softsign': 0.85,
}


# =============================================================================
# Evaluation Metrics
# =============================================================================

def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient.

    This is the primary metric used in the BC4D4 paper.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    float
        Pearson correlation coefficient [-1, 1]
    """
    return np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    dict
        Dictionary of metric names to values
    """
    return {
        'correlation': pearson_correlation(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'r_squared': r_squared(y_true, y_pred),
    }


# =============================================================================
# Model Evaluation
# =============================================================================

def load_model(checkpoint_path: str, device: torch.device = None) -> BC4D4:
    """
    Load a trained BC4D4 model from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to model checkpoint
    device : torch.device
        Device to load model on

    Returns
    -------
    BC4D4
        Loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = create_bc4d4_model(
        num_features=checkpoint['num_features'],
        activation=checkpoint['config'].get('activation', 'softsign'),
        dropout_rate=checkpoint['config'].get('dropout_rate', 0.1),
        device=str(device)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def evaluate_single_finger(model: BC4D4,
                          test_ecog: np.ndarray,
                          test_finger: np.ndarray,
                          finger_idx: int,
                          device: torch.device = None) -> Dict[str, float]:
    """
    Evaluate a single finger model.

    Parameters
    ----------
    model : BC4D4
        Trained model
    test_ecog : np.ndarray
        Test ECoG data, shape (N, features, 1)
    test_finger : np.ndarray
        Test finger data, shape (N, 5)
    finger_idx : int
        Which finger (0-4)
    device : torch.device
        Device to evaluate on

    Returns
    -------
    dict
        Dictionary of metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    # Convert to tensors
    ecog_tensor = torch.FloatTensor(test_ecog).to(device)
    target = test_finger[:, finger_idx]

    # Get predictions
    with torch.no_grad():
        predictions = model(ecog_tensor).cpu().numpy().flatten()

    # Compute metrics
    metrics = compute_all_metrics(target, predictions)
    metrics['finger'] = FINGER_NAMES[finger_idx]

    return metrics


def evaluate_all_fingers(model_dir: str,
                        test_ecog: np.ndarray,
                        test_finger: np.ndarray,
                        device: torch.device = None,
                        verbose: bool = True) -> Dict:
    """
    Evaluate all 5 finger models.

    Parameters
    ----------
    model_dir : str
        Directory containing model checkpoints
    test_ecog : np.ndarray
        Test ECoG data
    test_finger : np.ndarray
        Test finger data
    device : torch.device
        Device to evaluate on
    verbose : bool
        Whether to print results

    Returns
    -------
    dict
        Dictionary with per-finger and average metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = {
        'per_finger': {},
        'average': {},
    }

    correlations = []

    for finger_idx, finger_name in enumerate(FINGER_NAMES):
        # Load model
        model_path = os.path.join(model_dir, f'{finger_name.lower()}_model.pt')

        if not os.path.exists(model_path):
            if verbose:
                print(f"Warning: Model not found for {finger_name}: {model_path}")
            continue

        model = load_model(model_path, device)

        # Evaluate
        metrics = evaluate_single_finger(
            model, test_ecog, test_finger, finger_idx, device
        )

        results['per_finger'][finger_name] = metrics
        correlations.append(metrics['correlation'])

        if verbose:
            print(f"  {finger_name}: correlation = {metrics['correlation']:.4f}")

    # Compute averages
    results['average']['correlation'] = np.mean(correlations)
    results['average']['std'] = np.std(correlations)

    if verbose:
        print(f"\n  Average: {results['average']['correlation']:.4f} "
              f"(+/- {results['average']['std']:.4f})")

    return results


# =============================================================================
# Comparison with Paper Benchmarks
# =============================================================================

def compare_with_paper(results: Dict,
                       subject: int,
                       activation: str = 'softsign',
                       verbose: bool = True) -> pd.DataFrame:
    """
    Compare results with paper benchmarks.

    Parameters
    ----------
    results : dict
        Evaluation results from evaluate_all_fingers
    subject : int
        Subject number
    activation : str
        Which activation was used ('tanh' or 'softsign')
    verbose : bool
        Whether to print comparison

    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    paper_results = PAPER_RESULTS_SOFTSIGN if activation == 'softsign' else PAPER_RESULTS_TANH
    paper_subject = paper_results[subject]

    comparison_data = []

    for finger in FINGER_NAMES:
        our_corr = results['per_finger'].get(finger, {}).get('correlation', np.nan)
        paper_corr = paper_subject.get(finger, np.nan)
        diff = our_corr - paper_corr

        comparison_data.append({
            'Finger': finger,
            'Our Result': our_corr,
            'Paper Result': paper_corr,
            'Difference': diff,
            'Match': abs(diff) < 0.05  # Within 5% is considered a match
        })

    # Add average
    our_avg = results['average']['correlation']
    paper_avg = paper_subject['Average']
    comparison_data.append({
        'Finger': 'Average',
        'Our Result': our_avg,
        'Paper Result': paper_avg,
        'Difference': our_avg - paper_avg,
        'Match': abs(our_avg - paper_avg) < 0.05
    })

    df = pd.DataFrame(comparison_data)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Comparison with Paper - Subject {subject} ({activation})")
        print(f"{'='*60}")
        print(df.to_string(index=False))

        # Overall assessment
        n_matches = df['Match'].sum()
        print(f"\n{n_matches}/{len(df)} metrics within 5% of paper results")

    return df


def create_results_table(all_results: Dict[int, Dict],
                        activation: str = 'softsign') -> pd.DataFrame:
    """
    Create a summary table for all subjects.

    Parameters
    ----------
    all_results : dict
        Results for all subjects {subject_id: results}
    activation : str
        Activation function used

    Returns
    -------
    pd.DataFrame
        Summary table
    """
    paper_results = PAPER_RESULTS_SOFTSIGN if activation == 'softsign' else PAPER_RESULTS_TANH

    rows = []

    for subject in [1, 2, 3]:
        if subject not in all_results:
            continue

        results = all_results[subject]
        paper = paper_results[subject]

        row = {'Subject': f'Sub-{subject}'}

        for finger in FINGER_NAMES:
            our_corr = results['per_finger'].get(finger, {}).get('correlation', np.nan)
            row[finger] = our_corr

        row['Average'] = results['average']['correlation']
        row['Paper Avg'] = paper['Average']
        row['Diff'] = results['average']['correlation'] - paper['Average']

        rows.append(row)

    # Add overall row
    if rows:
        overall_our = np.mean([r['Average'] for r in rows])
        overall_paper = OVERALL_BENCHMARKS[f'BC4D4_{"Softsign" if activation == "softsign" else "Tanh"}']
        rows.append({
            'Subject': 'Overall',
            'Thumb': '',
            'Index': '',
            'Middle': '',
            'Ring': '',
            'Little': '',
            'Average': overall_our,
            'Paper Avg': overall_paper,
            'Diff': overall_our - overall_paper,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Visualization
# =============================================================================

def plot_predictions(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    finger_name: str,
                    save_path: str = None):
    """
    Plot predicted vs actual finger movements.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values
    finger_name : str
        Name of the finger
    save_path : str
        Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time series plot (first 500 samples)
    n_samples = min(500, len(y_true))
    axes[0].plot(y_true[:n_samples], label='Actual', alpha=0.7)
    axes[0].plot(y_pred[:n_samples], label='Predicted', alpha=0.7)
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Finger Movement')
    axes[0].set_title(f'{finger_name} - Time Series')
    axes[0].legend()

    # Scatter plot
    corr = pearson_correlation(y_true, y_pred)
    axes[1].scatter(y_true, y_pred, alpha=0.3, s=1)
    axes[1].plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()],
                 'r--', label='Perfect prediction')
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title(f'{finger_name} - Correlation: {corr:.4f}')
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    return fig, axes


def plot_comparison_bar(results: Dict,
                       subject: int,
                       activation: str = 'softsign',
                       save_path: str = None):
    """
    Create bar chart comparing results with paper.

    Parameters
    ----------
    results : dict
        Our evaluation results
    subject : int
        Subject number
    activation : str
        Activation function used
    save_path : str
        Optional path to save figure
    """
    paper_results = PAPER_RESULTS_SOFTSIGN if activation == 'softsign' else PAPER_RESULTS_TANH
    paper = paper_results[subject]

    fingers = FINGER_NAMES + ['Average']
    our_values = [results['per_finger'].get(f, {}).get('correlation', 0) for f in FINGER_NAMES]
    our_values.append(results['average']['correlation'])

    paper_values = [paper.get(f, 0) for f in FINGER_NAMES]
    paper_values.append(paper['Average'])

    x = np.arange(len(fingers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, our_values, width, label='Our Results', color='steelblue')
    bars2 = ax.bar(x + width/2, paper_values, width, label='Paper Results', color='coral')

    ax.set_ylabel('Correlation')
    ax.set_title(f'Subject {subject} - BC4D4 Results vs Paper ({activation})')
    ax.set_xticks(x)
    ax.set_xticklabels(fingers)
    ax.legend()
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")

    return fig, ax


def plot_method_comparison(all_results: Dict[int, Dict],
                          activation: str = 'softsign',
                          save_path: str = None):
    """
    Compare BC4D4 with FingerFlex baseline.

    Parameters
    ----------
    all_results : dict
        Our BC4D4 results for all subjects
    activation : str
        Activation function used
    save_path : str
        Optional path to save figure
    """
    subjects = [1, 2, 3]

    # Our BC4D4 results
    our_bc4d4 = [all_results.get(s, {}).get('average', {}).get('correlation', 0)
                 for s in subjects]

    # Paper BC4D4 results
    paper_results = PAPER_RESULTS_SOFTSIGN if activation == 'softsign' else PAPER_RESULTS_TANH
    paper_bc4d4 = [paper_results[s]['Average'] for s in subjects]

    # FingerFlex baseline
    fingerflex = [FINGERFLEX_RESULTS[s]['Average'] for s in subjects]

    x = np.arange(len(subjects))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, our_bc4d4, width, label=f'Our BC4D4 ({activation})',
                   color='steelblue')
    bars2 = ax.bar(x, paper_bc4d4, width, label=f'Paper BC4D4 ({activation})',
                   color='coral')
    bars3 = ax.bar(x + width, fingerflex, width, label='FingerFlex',
                   color='gray')

    ax.set_ylabel('Average Correlation')
    ax.set_title('Method Comparison: BC4D4 vs FingerFlex')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Subject {s}' for s in subjects])
    ax.legend()
    ax.set_ylim(0, 1)

    # Add improvement percentages
    for i, (our, ff) in enumerate(zip(our_bc4d4, fingerflex)):
        if ff > 0:
            improvement = (our - ff) / ff * 100
            ax.annotate(f'+{improvement:.1f}%',
                       xy=(x[i] - width, our),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha='center', fontsize=9, color='green')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Method comparison plot saved to: {save_path}")

    return fig, ax


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

def run_full_evaluation(run_dir: str,
                       data_path: str = None,
                       subject: int = 1,
                       verbose: bool = True) -> Dict:
    """
    Run full evaluation pipeline.

    Parameters
    ----------
    run_dir : str
        Directory containing trained models and preprocessed data
    data_path : str
        Path to test data (if not in run_dir)
    subject : int
        Subject number
    verbose : bool
        Whether to print results

    Returns
    -------
    dict
        Complete evaluation results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print("=" * 60)
        print(f"BC4D4 Evaluation - Subject {subject}")
        print("=" * 60)
        print(f"Run directory: {run_dir}")
        print(f"Device: {device}")

    # Load test data
    if data_path is None:
        data_path = os.path.join(run_dir, 'preprocessed')

    test_ecog = np.load(os.path.join(data_path, f'sub{subject}_test_ecog_bc4d4.npy'))
    test_finger = np.load(os.path.join(data_path, f'sub{subject}_test_finger_bc4d4.npy'))

    if verbose:
        print(f"\nTest data loaded:")
        print(f"  ECoG shape: {test_ecog.shape}")
        print(f"  Finger shape: {test_finger.shape}")

    # Load config to get activation
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    activation = config.get('activation', 'softsign')

    # Evaluate all fingers
    model_dir = os.path.join(run_dir, 'models')
    results = evaluate_all_fingers(
        model_dir=model_dir,
        test_ecog=test_ecog,
        test_finger=test_finger,
        device=device,
        verbose=verbose
    )

    # Compare with paper
    comparison_df = compare_with_paper(results, subject, activation, verbose)

    # Save results
    results_path = os.path.join(run_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'subject': subject,
            'activation': activation,
            'results': {
                'per_finger': {k: {kk: float(vv) for kk, vv in v.items() if isinstance(vv, (int, float))}
                              for k, v in results['per_finger'].items()},
                'average': {k: float(v) for k, v in results['average'].items()},
            }
        }, f, indent=2)

    if verbose:
        print(f"\nEvaluation results saved to: {results_path}")

    # Create comparison plot
    plot_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    plot_comparison_bar(
        results, subject, activation,
        save_path=os.path.join(plot_dir, f'comparison_sub{subject}.png')
    )
    plt.close()

    return {
        'results': results,
        'comparison': comparison_df,
        'activation': activation,
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='BC4D4 Evaluation')
    parser.add_argument('--run_dir', type=str, required=True,
                       help='Directory containing trained models')
    parser.add_argument('--subject', type=int, default=1, choices=[1, 2, 3],
                       help='Subject number')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to test data (optional)')

    args = parser.parse_args()

    run_full_evaluation(
        run_dir=args.run_dir,
        data_path=args.data_path,
        subject=args.subject,
        verbose=True
    )
