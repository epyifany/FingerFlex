"""
BC4D4 Model Architecture

Implements the CNN+DNN model from Jangir et al. (2025) paper:
"Harnessing the synergy of statistics and deep learning for BCI competition 4 dataset 4"

Architecture (from Table 3 in paper):
- CNN Block: 3 Conv1D layers with ReLU (no pooling)
- DNN Block: 6 Dense layers with Tanh/Softsign activation
- Output: Single value for one finger (train 5 separate models)

Key differences from FingerFlex:
1. Simpler sequential architecture (not U-Net)
2. No pooling layers (paper states pooling causes information loss)
3. Tanh/Softsign activation (not GELU)
4. Per-finger models (output=1, not 5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class Softsign(nn.Module):
    """
    Softsign activation function.

    f(x) = x / (1 + |x|)

    Properties:
    - Range: [-1, +1]
    - Grows polynomially (gentler than Tanh)
    - Avoids vanishing gradient better than Tanh
    - Paper shows ~5% better correlation than Tanh
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (1 + torch.abs(x))


class BC4D4(nn.Module):
    """
    BC4D4 CNN+DNN model for BCI finger movement regression.

    Architecture from paper Table 3:
    =========================================
    Layer | Type        | Output  | Activation
    =========================================
    0     | Conv1D      | 64      | ReLU
    1     | Conv1D      | 128     | ReLU
    2     | Conv1D      | 256     | ReLU
    3     | Flatten     | -       | -
    4     | Dense       | 1024    | Tanh/Softsign
    5     | Dropout     | -       | (0.1)
    6     | Dense       | 512     | Tanh/Softsign
    7     | Dense       | 256     | Tanh/Softsign
    8     | Dense       | 128     | Tanh/Softsign
    9     | Dense       | 64      | Tanh/Softsign
    10    | Dense       | 1       | Tanh/Softsign
    =========================================

    IMPORTANT: This model predicts ONE finger at a time.
    Train 5 separate models for all fingers.

    Parameters
    ----------
    num_features : int
        Number of ECoG electrodes/channels (62 for Subject 1)
    activation : str
        Activation for DNN block: 'tanh' or 'softsign'
    dropout_rate : float
        Dropout rate (paper uses 0.1)
    """

    def __init__(self,
                 num_features: int,
                 activation: Literal['tanh', 'softsign'] = 'softsign',
                 dropout_rate: float = 0.1):
        super(BC4D4, self).__init__()

        self.num_features = num_features
        self.activation_name = activation

        # =====================================================================
        # CNN Block (ReLU activation)
        # =====================================================================
        # Paper specifications:
        # - Kernel size: 3
        # - Stride: 1
        # - NO padding (paper doesn't mention padding)
        # - NO pooling (paper explicitly states pooling causes information loss)

        # Layer 0: Conv1D(1 -> 64)
        # Input: (batch, 1, num_features)
        # Output: (batch, 64, num_features - 2)
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0  # No padding
        )

        # Layer 1: Conv1D(64 -> 128)
        # Output: (batch, 128, num_features - 4)
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=0
        )

        # Layer 2: Conv1D(128 -> 256)
        # Output: (batch, 256, num_features - 6)
        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=0
        )

        # Calculate flatten size after CNN block
        # After 3 conv layers with kernel=3, no padding:
        # output_length = num_features - 2*3 = num_features - 6
        self.flatten_size = (num_features - 6) * 256

        # =====================================================================
        # DNN Block (Tanh or Softsign activation)
        # =====================================================================

        # Select activation function based on parameter
        if activation == 'tanh':
            self.dense_activation = nn.Tanh()
        elif activation == 'softsign':
            self.dense_activation = Softsign()
        else:
            raise ValueError(f"Unknown activation: {activation}. Use 'tanh' or 'softsign'")

        # Layer 4: Dense(flatten -> 1024)
        self.fc1 = nn.Linear(self.flatten_size, 1024)

        # Layer 5: Dropout(0.1)
        self.dropout = nn.Dropout(dropout_rate)

        # Layer 6: Dense(1024 -> 512)
        self.fc2 = nn.Linear(1024, 512)

        # Layer 7: Dense(512 -> 256)
        self.fc3 = nn.Linear(512, 256)

        # Layer 8: Dense(256 -> 128)
        self.fc4 = nn.Linear(256, 128)

        # Layer 9: Dense(128 -> 64)
        self.fc5 = nn.Linear(128, 64)

        # Layer 10: Dense(64 -> 1) - OUTPUT (one finger)
        self.fc6 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, num_features, 1)
            This is raw ECoG data, NOT wavelet spectrograms

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, 1)
            Predicted finger movement value
        """
        # Input shape: (batch, num_features, 1)
        # Permute to: (batch, 1, num_features) for Conv1d
        x = x.permute(0, 2, 1)

        # =====================================================================
        # CNN Block with ReLU
        # =====================================================================
        x = F.relu(self.conv1(x))  # (batch, 64, num_features-2)
        x = F.relu(self.conv2(x))  # (batch, 128, num_features-4)
        x = F.relu(self.conv3(x))  # (batch, 256, num_features-6)

        # =====================================================================
        # Flatten
        # =====================================================================
        x = x.view(x.size(0), -1)  # (batch, 256 * (num_features-6))

        # =====================================================================
        # DNN Block with Tanh/Softsign
        # =====================================================================
        x = self.dense_activation(self.fc1(x))  # (batch, 1024)
        x = self.dropout(x)
        x = self.dense_activation(self.fc2(x))  # (batch, 512)
        x = self.dense_activation(self.fc3(x))  # (batch, 256)
        x = self.dense_activation(self.fc4(x))  # (batch, 128)
        x = self.dense_activation(self.fc5(x))  # (batch, 64)
        x = self.dense_activation(self.fc6(x))  # (batch, 1)

        return x

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_architecture(self):
        """Print detailed architecture summary."""
        print("=" * 60)
        print("BC4D4 Model Architecture")
        print("=" * 60)
        print(f"\nInput: ({self.num_features}, 1) - Raw ECoG signals")
        print(f"Activation: {self.activation_name}")
        print(f"\n--- CNN Block (ReLU) ---")
        print(f"Conv1: 1 -> 64, kernel=3, no padding")
        print(f"Conv2: 64 -> 128, kernel=3, no padding")
        print(f"Conv3: 128 -> 256, kernel=3, no padding")
        print(f"Flatten: {self.flatten_size}")
        print(f"\n--- DNN Block ({self.activation_name}) ---")
        print(f"FC1: {self.flatten_size} -> 1024")
        print(f"Dropout: 0.1")
        print(f"FC2: 1024 -> 512")
        print(f"FC3: 512 -> 256")
        print(f"FC4: 256 -> 128")
        print(f"FC5: 128 -> 64")
        print(f"FC6: 64 -> 1 (OUTPUT)")
        print(f"\nTotal Parameters: {self.count_parameters():,}")
        print("=" * 60)


class BC4D4Ensemble(nn.Module):
    """
    Ensemble of 5 BC4D4 models, one for each finger.

    This wraps 5 separate BC4D4 models to predict all fingers,
    but each model is trained independently on a single finger.

    Useful for inference when you want predictions for all fingers.
    """

    def __init__(self,
                 num_features: int,
                 activation: Literal['tanh', 'softsign'] = 'softsign',
                 dropout_rate: float = 0.1):
        super(BC4D4Ensemble, self).__init__()

        self.finger_names = ['thumb', 'index', 'middle', 'ring', 'little']

        # Create 5 separate models
        self.models = nn.ModuleDict({
            name: BC4D4(num_features, activation, dropout_rate)
            for name in self.finger_names
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for all 5 fingers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, num_features, 1)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, 5)
            Predictions for all 5 fingers
        """
        outputs = []
        for name in self.finger_names:
            out = self.models[name](x)
            outputs.append(out)

        # Concatenate: (batch, 5)
        return torch.cat(outputs, dim=1)

    def get_finger_model(self, finger_idx: int) -> BC4D4:
        """Get the model for a specific finger."""
        return self.models[self.finger_names[finger_idx]]


def create_bc4d4_model(num_features: int,
                       activation: str = 'softsign',
                       dropout_rate: float = 0.1,
                       device: str = 'cpu') -> BC4D4:
    """
    Factory function to create a BC4D4 model.

    Parameters
    ----------
    num_features : int
        Number of ECoG electrodes (62 for Subject 1, 48 for Subject 2, 64 for Subject 3)
    activation : str
        'tanh' or 'softsign' (softsign recommended for best results)
    dropout_rate : float
        Dropout rate (default 0.1 from paper)
    device : str
        Device to place model on ('cpu' or 'cuda')

    Returns
    -------
    BC4D4
        Initialized model on specified device
    """
    model = BC4D4(
        num_features=num_features,
        activation=activation,
        dropout_rate=dropout_rate
    )
    return model.to(device)


def verify_model_architecture():
    """
    Verify model architecture matches paper specifications.

    Expected parameter counts from paper Table 3:
    - Conv1: (3×1×64) + 64 = 256
    - Conv2: (3×64×128) + 128 = 24,704
    - Conv3: (3×128×256) + 256 = 98,560
    - FC1: depends on num_features
    - FC2: (1024×512) + 512 = 524,800
    - FC3: (512×256) + 256 = 131,328
    - FC4: (256×128) + 128 = 32,896
    - FC5: (128×64) + 64 = 8,256
    - FC6: (64×1) + 1 = 65
    """
    print("Verifying BC4D4 model architecture...")
    print()

    # Test with Subject 1 electrode count
    model = BC4D4(num_features=62, activation='softsign')

    # Print architecture
    model.print_architecture()

    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(32, 62, 1)  # (batch=32, features=62, 1)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (32, 1), f"Expected output shape (32, 1), got {y.shape}"
    print("Forward pass successful!")

    # Verify layer parameter counts
    print("\nVerifying layer parameter counts...")

    # Conv layers
    conv1_params = sum(p.numel() for p in model.conv1.parameters())
    conv2_params = sum(p.numel() for p in model.conv2.parameters())
    conv3_params = sum(p.numel() for p in model.conv3.parameters())

    print(f"Conv1 params: {conv1_params} (expected: 256)")
    print(f"Conv2 params: {conv2_params} (expected: 24,704)")
    print(f"Conv3 params: {conv3_params} (expected: 98,560)")

    # Verify conv layer counts match paper
    assert conv1_params == 256, f"Conv1 param mismatch: {conv1_params} != 256"
    assert conv2_params == 24704, f"Conv2 param mismatch: {conv2_params} != 24704"
    assert conv3_params == 98560, f"Conv3 param mismatch: {conv3_params} != 98560"

    # FC layers (excluding FC1 which depends on num_features)
    fc2_params = sum(p.numel() for p in model.fc2.parameters())
    fc3_params = sum(p.numel() for p in model.fc3.parameters())
    fc4_params = sum(p.numel() for p in model.fc4.parameters())
    fc5_params = sum(p.numel() for p in model.fc5.parameters())
    fc6_params = sum(p.numel() for p in model.fc6.parameters())

    print(f"FC2 params: {fc2_params} (expected: 524,800)")
    print(f"FC3 params: {fc3_params} (expected: 131,328)")
    print(f"FC4 params: {fc4_params} (expected: 32,896)")
    print(f"FC5 params: {fc5_params} (expected: 8,256)")
    print(f"FC6 params: {fc6_params} (expected: 65)")

    # Verify FC layer counts match paper
    assert fc2_params == 524800, f"FC2 param mismatch: {fc2_params} != 524800"
    assert fc3_params == 131328, f"FC3 param mismatch: {fc3_params} != 131328"
    assert fc4_params == 32896, f"FC4 param mismatch: {fc4_params} != 32896"
    assert fc5_params == 8256, f"FC5 param mismatch: {fc5_params} != 8256"
    assert fc6_params == 65, f"FC6 param mismatch: {fc6_params} != 65"

    print("\nAll parameter counts match paper specifications!")
    print("Model architecture verification PASSED!")

    return model


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Run architecture verification
    verify_model_architecture()

    print("\n" + "=" * 60)
    print("Testing all subject configurations...")
    print("=" * 60)

    subject_electrodes = {1: 62, 2: 48, 3: 64}

    for subject, num_features in subject_electrodes.items():
        print(f"\nSubject {subject} ({num_features} electrodes):")
        model = BC4D4(num_features=num_features, activation='softsign')
        x = torch.randn(16, num_features, 1)
        y = model(x)
        print(f"  Input: {x.shape} -> Output: {y.shape}")
        print(f"  Parameters: {model.count_parameters():,}")
