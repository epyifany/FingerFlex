"""
BC4D4 Training Script

Trains the BC4D4 model following Jangir et al. (2025) methodology.

Key training characteristics:
1. Train 5 SEPARATE models (one per finger)
2. Use MSE loss for regression
3. Softsign activation for best results
4. Per-subject models (different electrode counts)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

from BC4D4_model import BC4D4, create_bc4d4_model
from BC4D4_preprocessing import (
    preprocess_bc4d4,
    load_preprocessed_data,
    save_preprocessed_data,
    FINGER_NAMES,
    SUBJECT_ELECTRODES
)


# =============================================================================
# Dataset Classes
# =============================================================================

class BC4D4Dataset(Dataset):
    """
    PyTorch Dataset for BC4D4 training.

    Provides (ECoG, finger_value) pairs for a single finger.
    """

    def __init__(self,
                 ecog_data: np.ndarray,
                 finger_data: np.ndarray,
                 finger_idx: int = 0):
        """
        Parameters
        ----------
        ecog_data : np.ndarray
            ECoG data, shape (N, num_features, 1)
        finger_data : np.ndarray
            Finger movement data, shape (N, 5)
        finger_idx : int
            Which finger to use (0=thumb, ..., 4=little)
        """
        self.ecog = torch.FloatTensor(ecog_data)
        self.finger = torch.FloatTensor(finger_data[:, finger_idx:finger_idx+1])

    def __len__(self):
        return len(self.ecog)

    def __getitem__(self, idx):
        return self.ecog[idx], self.finger[idx]


def create_dataloaders(train_ecog: np.ndarray,
                       train_finger: np.ndarray,
                       test_ecog: np.ndarray,
                       test_finger: np.ndarray,
                       finger_idx: int,
                       batch_size: int = 64,
                       val_split: float = 0.1,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for a single finger.

    Parameters
    ----------
    train_ecog : np.ndarray
        Training ECoG data
    train_finger : np.ndarray
        Training finger data (all 5 fingers)
    test_ecog : np.ndarray
        Test ECoG data
    test_finger : np.ndarray
        Test finger data (all 5 fingers)
    finger_idx : int
        Which finger to train on
    batch_size : int
        Batch size
    val_split : float
        Fraction of training data for validation
    num_workers : int
        Number of dataloader workers

    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader)
    """
    # Split training data into train/val
    n_train = len(train_ecog)
    n_val = int(n_train * val_split)
    indices = np.random.permutation(n_train)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    # Create datasets
    train_dataset = BC4D4Dataset(
        train_ecog[train_indices],
        train_finger[train_indices],
        finger_idx
    )
    val_dataset = BC4D4Dataset(
        train_ecog[val_indices],
        train_finger[val_indices],
        finger_idx
    )
    test_dataset = BC4D4Dataset(
        test_ecog,
        test_finger,
        finger_idx
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model: BC4D4,
                dataloader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device) -> float:
    """
    Train for one epoch.

    Returns
    -------
    float
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for ecog, finger in dataloader:
        ecog = ecog.to(device)
        finger = finger.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(ecog)
        loss = criterion(output, finger)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model: BC4D4,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """
    Evaluate model on a dataset.

    Returns
    -------
    tuple
        (average_loss, pearson_correlation)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for ecog, finger in dataloader:
            ecog = ecog.to(device)
            finger = finger.to(device)

            output = model(ecog)
            loss = criterion(output, finger)

            total_loss += loss.item()
            n_batches += 1

            all_preds.append(output.cpu().numpy())
            all_targets.append(finger.cpu().numpy())

    # Compute Pearson correlation
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    correlation = np.corrcoef(all_preds, all_targets)[0, 1]

    avg_loss = total_loss / n_batches

    return avg_loss, correlation


def train_single_finger(train_ecog: np.ndarray,
                        train_finger: np.ndarray,
                        test_ecog: np.ndarray,
                        test_finger: np.ndarray,
                        finger_idx: int,
                        num_features: int,
                        config: Dict,
                        device: torch.device,
                        save_dir: str = None,
                        verbose: bool = True) -> Tuple[BC4D4, Dict]:
    """
    Train a BC4D4 model for a single finger.

    Parameters
    ----------
    train_ecog : np.ndarray
        Training ECoG data
    train_finger : np.ndarray
        Training finger data
    test_ecog : np.ndarray
        Test ECoG data
    test_finger : np.ndarray
        Test finger data
    finger_idx : int
        Which finger (0-4)
    num_features : int
        Number of electrodes
    config : dict
        Training configuration
    device : torch.device
        Device to train on
    save_dir : str
        Directory to save checkpoints
    verbose : bool
        Whether to print progress

    Returns
    -------
    tuple
        (trained_model, training_history)
    """
    finger_name = FINGER_NAMES[finger_idx]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training model for {finger_name} finger")
        print(f"{'='*60}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ecog, train_finger,
        test_ecog, test_finger,
        finger_idx=finger_idx,
        batch_size=config.get('batch_size', 64),
        val_split=config.get('val_split', 0.1)
    )

    # Create model
    model = create_bc4d4_model(
        num_features=num_features,
        activation=config.get('activation', 'softsign'),
        dropout_rate=config.get('dropout_rate', 0.1),
        device=str(device)
    )

    # Loss function (MSE for regression)
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 0)
    )

    # Learning rate scheduler (optional)
    scheduler = None
    if config.get('use_scheduler', False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=verbose
        )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_corr': [],
        'test_loss': None,
        'test_corr': None,
        'best_epoch': 0,
        'best_val_corr': 0,
    }

    # Best model tracking
    best_val_corr = -1
    best_model_state = None
    patience_counter = 0
    patience = config.get('patience', 20)

    num_epochs = config.get('num_epochs', 100)

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_corr = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_corr'].append(val_corr)

        # Check for best model
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_model_state = model.state_dict().copy()
            history['best_epoch'] = epoch
            history['best_val_corr'] = val_corr
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Val Corr: {val_corr:.4f}")

        # Early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final test evaluation
    test_loss, test_corr = evaluate(model, test_loader, criterion, device)
    history['test_loss'] = test_loss
    history['test_corr'] = test_corr

    if verbose:
        print(f"\n{finger_name} Results:")
        print(f"  Best Val Correlation: {history['best_val_corr']:.4f} (epoch {history['best_epoch']+1})")
        print(f"  Test Correlation: {test_corr:.4f}")

    # Save checkpoint
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, f'{finger_name.lower()}_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'num_features': num_features,
            'finger_idx': finger_idx,
            'history': history,
        }, checkpoint_path)
        if verbose:
            print(f"  Model saved to: {checkpoint_path}")

    return model, history


def train_all_fingers(data: Dict[str, np.ndarray],
                      num_features: int,
                      config: Dict,
                      device: torch.device,
                      save_dir: str = None,
                      verbose: bool = True) -> Tuple[List[BC4D4], Dict]:
    """
    Train BC4D4 models for all 5 fingers.

    Parameters
    ----------
    data : dict
        Preprocessed data dictionary
    num_features : int
        Number of electrodes
    config : dict
        Training configuration
    device : torch.device
        Device to train on
    save_dir : str
        Directory to save models
    verbose : bool
        Whether to print progress

    Returns
    -------
    tuple
        (list of 5 models, results dictionary)
    """
    if verbose:
        print("\n" + "=" * 60)
        print("BC4D4 Training - All 5 Fingers")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"Number of features: {num_features}")
        print(f"Activation: {config.get('activation', 'softsign')}")
        print(f"Batch size: {config.get('batch_size', 64)}")
        print(f"Learning rate: {config.get('learning_rate', 1e-3)}")
        print(f"Epochs: {config.get('num_epochs', 100)}")

    models = []
    results = {
        'finger_correlations': {},
        'average_correlation': 0,
        'config': config,
    }

    for finger_idx in range(5):
        model, history = train_single_finger(
            train_ecog=data['train_ecog'],
            train_finger=data['train_finger'],
            test_ecog=data['test_ecog'],
            test_finger=data['test_finger'],
            finger_idx=finger_idx,
            num_features=num_features,
            config=config,
            device=device,
            save_dir=save_dir,
            verbose=verbose
        )
        models.append(model)
        results['finger_correlations'][FINGER_NAMES[finger_idx]] = history['test_corr']

    # Calculate average correlation
    avg_corr = np.mean(list(results['finger_correlations'].values()))
    results['average_correlation'] = avg_corr

    if verbose:
        print("\n" + "=" * 60)
        print("Final Results")
        print("=" * 60)
        for finger, corr in results['finger_correlations'].items():
            print(f"  {finger}: {corr:.4f}")
        print(f"  Average: {avg_corr:.4f}")
        print("=" * 60)

    # Save results summary
    if save_dir:
        results_path = os.path.join(save_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"\nResults saved to: {results_path}")

    return models, results


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    # Model
    'activation': 'softsign',  # 'tanh' or 'softsign' (paper recommends softsign)
    'dropout_rate': 0.1,

    # Training
    'num_epochs': 100,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'weight_decay': 0,
    'val_split': 0.1,

    # Early stopping
    'patience': 20,
    'use_scheduler': True,

    # Preprocessing
    'contamination': 'auto',  # Isolation Forest contamination
}


# =============================================================================
# Main Entry Point
# =============================================================================

def main(data_path: str = './data/pure_data',
         output_path: str = './outputs/bc4d4',
         subject: int = 1,
         config: Dict = None,
         preprocess: bool = True,
         verbose: bool = True):
    """
    Main training pipeline.

    Parameters
    ----------
    data_path : str
        Path to raw data (if preprocess=True) or preprocessed data
    output_path : str
        Path to save outputs
    subject : int
        Subject number (1, 2, or 3)
    config : dict
        Training configuration (defaults to DEFAULT_CONFIG)
    preprocess : bool
        Whether to run preprocessing (if False, expects preprocessed data)
    verbose : bool
        Whether to print progress
    """
    # Set up configuration
    if config is None:
        config = DEFAULT_CONFIG.copy()

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(output_path, f'sub{subject}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print("=" * 60)
        print(f"BC4D4 Training Pipeline - Subject {subject}")
        print("=" * 60)
        print(f"Data path: {data_path}")
        print(f"Output path: {run_dir}")
        print(f"Device: {device}")

    # Get data
    if preprocess:
        if verbose:
            print("\n[Preprocessing] Running BC4D4 preprocessing pipeline...")

        data = preprocess_bc4d4(
            data_path=data_path,
            subject=subject,
            contamination=config.get('contamination', 'auto'),
            save_plots=True,
            output_dir=run_dir,
            verbose=verbose
        )

        # Save preprocessed data
        preprocessed_dir = os.path.join(run_dir, 'preprocessed')
        save_preprocessed_data(data, preprocessed_dir, subject)
    else:
        if verbose:
            print("\n[Loading] Loading preprocessed data...")
        data = load_preprocessed_data(data_path, subject)

    # Get number of features for this subject
    num_features = SUBJECT_ELECTRODES[subject]

    # Train all fingers
    models, results = train_all_fingers(
        data=data,
        num_features=num_features,
        config=config,
        device=device,
        save_dir=os.path.join(run_dir, 'models'),
        verbose=verbose
    )

    if verbose:
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"All outputs saved to: {run_dir}")
        print("=" * 60)

    return models, results, run_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='BC4D4 Training')
    parser.add_argument('--data_path', type=str, default='./data/pure_data',
                       help='Path to data directory')
    parser.add_argument('--output_path', type=str, default='./outputs/bc4d4',
                       help='Path to save outputs')
    parser.add_argument('--subject', type=int, default=1, choices=[1, 2, 3],
                       help='Subject number')
    parser.add_argument('--activation', type=str, default='softsign',
                       choices=['tanh', 'softsign'],
                       help='DNN activation function')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--no_preprocess', action='store_true',
                       help='Skip preprocessing (use pre-saved data)')

    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config['activation'] = args.activation
    config['num_epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr

    main(
        data_path=args.data_path,
        output_path=args.output_path,
        subject=args.subject,
        config=config,
        preprocess=not args.no_preprocess,
        verbose=True
    )
