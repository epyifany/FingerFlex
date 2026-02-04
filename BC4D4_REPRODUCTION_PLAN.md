# BC4D4 Reproduction Plan

## Paper Reference
**Title:** "Harnessing the synergy of statistics and deep learning for BCI competition 4 dataset 4: a novel approach"
**Authors:** Gauttam Jangir, Nisheeth Joshi, Gaurav Purohit
**Published:** Brain Informatics, February 15, 2025
**Claimed Result:** 0.85 correlation (1.25x better than FingerFlex's 0.67)

---

## Executive Summary

BC4D4 achieves superior performance through:
1. **Isolation Forest outlier removal** (not used in FingerFlex)
2. **Working on RAW ECoG signals** (not wavelet spectrograms)
3. **Tanh/Softsign activation** (instead of GELU)
4. **Simpler CNN+DNN architecture** (instead of U-Net)
5. **Per-finger regression models** (output=1, not 5)

---

## PHASE 1: Data Understanding & Statistical Analysis

### Task 1.1: Load and Analyze Raw Dataset
- [ ] Load BCI Competition 4 Dataset 4 from `sub1_comp.mat`
- [ ] Understand data structure:
  - Subject 1: 62 electrodes/features
  - Subject 2: 48 electrodes/features
  - Subject 3: 64 electrodes/features
- [ ] Training data: `train_data` (ECoG), `train_dg` (finger flexion)
- [ ] Test data: `test_data` (ECoG), `test_dg` (finger flexion from `sub1_testlabels.mat`)

### Task 1.2: Statistical Analysis (Box Plots)
- [ ] Create box plots for all 5 fingers for each subject
- [ ] Calculate descriptive statistics (mean, std, min, 25%, 50%, 75%, max)
- [ ] Document outliers identified via box plot analysis
- [ ] **Expected finding:** Data has outliers, abrupt distribution (mean far from min/max)

### Task 1.3: Verify Data Distribution
- [ ] Plot histograms for each finger before preprocessing
- [ ] Confirm data range extends from approximately -1 to +5/+8 (with outliers)
- [ ] Document the non-Gaussian nature of raw data

---

## PHASE 2: Data Preprocessing with Isolation Forest

### Task 2.1: Implement Isolation Forest Outlier Removal
**CRITICAL:** This is the key innovation from BC4D4

```python
from sklearn.ensemble import IsolationForest

# Apply Isolation Forest to remove outliers
# contamination parameter needs heuristic tuning
isolation_forest = IsolationForest(
    contamination='auto',  # or specific value like 0.1
    random_state=42,
    n_estimators=100
)
```

- [ ] Apply Isolation Forest to finger movement data (train_dg)
- [ ] Apply Isolation Forest to ECoG data (train_data)
- [ ] Tune contamination level heuristically for best results
- [ ] **Goal:** Transform data from abrupt distribution to near-Gaussian

### Task 2.2: Validate Preprocessing Results
Expected results after Isolation Forest (from paper, Subject 1):

| Finger | Mean (before) | Mean (after) | Min (after) | Max (after) |
|--------|---------------|--------------|-------------|-------------|
| Thumb  | -0.01         | -0.35        | -0.78       | +0.71       |
| Index  | -0.01         | -0.29        | -1.09       | +0.83       |
| Middle | 0.00          | -0.31        | -0.84       | +0.94       |
| Ring   | -0.01         | -0.30        | -1.00       | +0.94       |
| Little | 0.01          | -0.29        | -0.87       | +0.90       |

- [ ] Verify data now falls approximately in range [-1, +1]
- [ ] Create box plots after preprocessing (should show minimal outliers)
- [ ] Create histograms showing near-Gaussian distribution

### Task 2.3: Data Normalization
- [ ] Normalize ECoG signals (z-score normalization)
- [ ] **Note:** BC4D4 does NOT use wavelet spectrograms - works on raw signals
- [ ] **Note:** BC4D4 does NOT use FingerFlex's bandpass filtering approach

---

## PHASE 3: BC4D4 Model Architecture Implementation

### Task 3.1: Input Data Preparation
**CRITICAL DIFFERENCE FROM FINGERFLEX:**
- FingerFlex: Input shape `(batch, 62 electrodes, 40 wavelets, 256 time)`
- BC4D4: Input shape `(batch, num_features, 1)` - raw signals, no spectrograms

```python
# BC4D4 Input format
# For Subject 1: (batch_size, 62, 1)
# For Subject 2: (batch_size, 48, 1)
# For Subject 3: (batch_size, 64, 1)
```

- [ ] Reshape data to match BC4D4 input requirements
- [ ] Create windowed samples from continuous data
- [ ] Determine optimal window/batch size ("ip" in paper)

### Task 3.2: Implement CNN Block
```python
# CNN Block Architecture (from Table 3 in paper)
# Layer 0: Conv1D + ReLU
#   - Input: (num_features, 1)
#   - Kernel: 3×1×64 (kernel_size=3, in_channels=1, out_channels=64)
#   - Output: (num_features, 64)
#   - Params: (3×1×64) + 64 = 256

# Layer 1: Conv1D + ReLU
#   - Input: (num_features, 64)
#   - Kernel: 3×64×128
#   - Output: (num_features, 128)
#   - Params: (3×64×128) + 128 = 24,704

# Layer 2: Conv1D + ReLU
#   - Input: (num_features, 128)
#   - Kernel: 3×128×256
#   - Output: (num_features, 256)
#   - Params: (3×128×256) + 256 = 98,560

# Layer 3: Flatten
#   - Output: num_features × 256
```

**IMPORTANT CNN SPECIFICATIONS:**
- [ ] Kernel size: 3 (1D convolution)
- [ ] Stride: 1
- [ ] **NO PADDING** (paper doesn't mention padding)
- [ ] **NO POOLING LAYERS** (paper explicitly states pooling causes information loss)
- [ ] Activation: ReLU for CNN layers

### Task 3.3: Implement DNN Block
```python
# DNN Block Architecture
# Layer 4: Dense(1024) + Tanh/Softsign
#   - Input: Flatten output (num_features × 256)
#   - Output: 1024
#   - Params: (256×1024) + 1024 = 262,144

# Layer 5: Dropout(0.1)
#   - Rate: 0.1 (10%)

# Layer 6: Dense(512) + Tanh/Softsign
#   - Params: (1024×512) + 512 = 524,800

# Layer 7: Dense(256) + Tanh/Softsign
#   - Params: (512×256) + 256 = 131,328

# Layer 8: Dense(128) + Tanh/Softsign
#   - Params: (256×128) + 128 = 32,896

# Layer 9: Dense(64) + Tanh/Softsign
#   - Params: (128×64) + 64 = 8,256

# Layer 10: Dense(1) + Tanh/Softsign (OUTPUT)
#   - Params: (64×1) + 1 = 65
```

### Task 3.4: Activation Function Selection
**KEY INSIGHT FROM PAPER:**
- Data after Isolation Forest has dual polarity (positive AND negative values)
- Data follows near-Gaussian distribution
- Range approximately [-1, +1]

**Activation Function Comparison:**
| Function   | Range      | Suitable? | Reason |
|------------|------------|-----------|--------|
| Sigmoid    | [0, 1]     | NO        | Doesn't cover negative values |
| ReLU       | [0, +∞)    | NO        | Doesn't cover negative values |
| LeakyReLU  | (-∞, +∞)   | NO        | Linear, not suitable for non-linear patterns |
| **Tanh**   | [-1, +1]   | **YES**   | Covers both polarities, non-linear |
| **Softsign**| [-1, +1]  | **YES**   | Covers both polarities, gentler gradient |

- [ ] Implement model with Tanh activation (baseline)
- [ ] Implement model with Softsign activation (best performance)

**Softsign vs Tanh:**
- Softsign: `f(x) = x / (1 + |x|)` - grows polynomially
- Tanh: `f(x) = (2 / (1 + e^(-2x))) - 1` - grows exponentially
- Softsign has gentler non-linearity, avoids vanishing gradient
- **Softsign achieves ~5% better correlation than Tanh**

### Task 3.5: Complete Model Implementation
```python
import torch
import torch.nn as nn

class BC4D4(nn.Module):
    def __init__(self, num_features, activation='tanh'):
        super(BC4D4, self).__init__()

        # Select activation function
        if activation == 'tanh':
            self.dense_activation = nn.Tanh()
        elif activation == 'softsign':
            self.dense_activation = nn.Softsign()

        # CNN Block
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1)
        self.relu = nn.ReLU()

        # Calculate flatten size after convolutions
        # After 3 conv layers with kernel=3, no padding:
        # output_length = input_length - 2*3 = input_length - 6
        flatten_size = (num_features - 6) * 256

        # DNN Block
        self.fc1 = nn.Linear(flatten_size, 1024)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch, num_features, 1)
        x = x.permute(0, 2, 1)  # -> (batch, 1, num_features)

        # CNN Block
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # DNN Block
        x = self.dense_activation(self.fc1(x))
        x = self.dropout(x)
        x = self.dense_activation(self.fc2(x))
        x = self.dense_activation(self.fc3(x))
        x = self.dense_activation(self.fc4(x))
        x = self.dense_activation(self.fc5(x))
        x = self.dense_activation(self.fc6(x))

        return x
```

---

## PHASE 4: Training Configuration

### Task 4.1: Training Setup
**CRITICAL:** BC4D4 trains a SEPARATE model for EACH finger

- [ ] Create 5 separate models (one per finger: thumb, index, middle, ring, little)
- [ ] Each model outputs a single regression value
- [ ] Loss function: MSE (Mean Squared Error) for regression
- [ ] Optimizer: Adam (standard choice, paper doesn't specify)

### Task 4.2: Data Splitting
- [ ] Use training data from `train_data` / `train_dg`
- [ ] Use test data from `test_data` / `test_dg` (from `sub1_testlabels.mat`)
- [ ] Create validation split if needed

### Task 4.3: Training Loop
```python
# Pseudocode for training
for finger_idx in range(5):  # thumb, index, middle, ring, little
    model = BC4D4(num_features=62, activation='softsign')

    # Train on single finger data
    y_train = finger_data[:, finger_idx]

    for epoch in range(num_epochs):
        # Forward pass
        y_pred = model(X_train)
        loss = mse_loss(y_pred, y_train)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Task 4.4: Hyperparameters to Tune
- [ ] Learning rate
- [ ] Batch size / window size ("ip" in paper)
- [ ] Number of epochs
- [ ] Dropout rate (paper uses 0.1)
- [ ] Isolation Forest contamination level

---

## PHASE 5: Evaluation & Metrics

### Task 5.1: Correlation Calculation
Primary metric: **Pearson Correlation Coefficient**

```python
import numpy as np

def pearson_correlation(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]
```

### Task 5.2: Expected Results (Target Benchmarks)

**With Tanh Activation:**
| Subject | Thumb | Index | Middle | Ring  | Little | **Avg** |
|---------|-------|-------|--------|-------|--------|---------|
| Sub-1   | 0.89  | 0.87  | 0.85   | 0.86  | 0.88   | **0.87**|
| Sub-2   | 0.79  | 0.66  | 0.45   | 0.70  | 0.78   | **0.68**|
| Sub-3   | 0.89  | 0.86  | 0.90   | 0.93  | 0.83   | **0.88**|
| **Overall** |   |       |        |       |        | **0.81**|

**With Softsign Activation (BEST):**
| Subject | Thumb | Index | Middle | Ring  | Little | **Avg** |
|---------|-------|-------|--------|-------|--------|---------|
| Sub-1   | 0.88  | 0.86  | 0.84   | 0.86  | 0.86   | **0.86**|
| Sub-2   | 0.82  | 0.81  | 0.79   | 0.83  | 0.80   | **0.81**|
| Sub-3   | 0.91  | 0.85  | 0.91   | 0.93  | 0.90   | **0.90**|
| **Overall** |   |       |        |       |        | **0.85**|

### Task 5.3: Comparison with FingerFlex

| Model | Subject 1 | Subject 2 | Subject 3 | **Overall** |
|-------|-----------|-----------|-----------|-------------|
| FingerFlex (GELU) | 0.66 | 0.62 | 0.74 | **0.67** |
| BC4D4 (Tanh) | 0.87 | 0.67 | 0.88 | **0.81** |
| BC4D4 (Softsign) | 0.86 | 0.81 | 0.90 | **0.85** |

---

## PHASE 6: Implementation Checklist

### 6.1: Files to Create/Modify
- [ ] `BC4D4_preprocessing.py` - Isolation Forest preprocessing
- [ ] `BC4D4_model.py` - Model architecture
- [ ] `BC4D4_train.py` - Training script
- [ ] `BC4D4_evaluate.py` - Evaluation and metrics
- [ ] `BC4D4_notebook.ipynb` - Complete pipeline notebook

### 6.2: Key Differences from FingerFlex to Implement

| Aspect | FingerFlex | BC4D4 |
|--------|------------|-------|
| Input | Wavelet spectrograms (62×40×time) | Raw ECoG (features×1) |
| Preprocessing | Bandpass filter, notch, CAR, wavelets | Isolation Forest outlier removal |
| Architecture | U-Net encoder-decoder | CNN + DNN (sequential) |
| Activation | GELU | Tanh or Softsign |
| Output | 5 fingers simultaneously | 1 finger per model |
| Pooling | MaxPool1d | None (explicitly avoided) |
| Skip connections | Yes | No |

### 6.3: Potential Issues to Watch

1. **Window/Batch Size:** Paper uses "ip" variable but doesn't specify exact value
2. **Contamination Level:** Isolation Forest contamination needs tuning
3. **Data Format:** Ensure correct reshaping from FingerFlex format
4. **Per-Finger Training:** Need to train 5 separate models
5. **Subject-Specific Models:** May need different models per subject

---

## PHASE 7: Step-by-Step Execution Plan

### Step 1: Statistical Analysis
```
1. Load raw data from .mat files
2. Create box plots for all fingers (before preprocessing)
3. Calculate descriptive statistics
4. Document outliers
```

### Step 2: Isolation Forest Preprocessing
```
1. Implement Isolation Forest on finger movement data
2. Tune contamination parameter
3. Apply to ECoG data
4. Verify near-Gaussian distribution
5. Create box plots (after preprocessing)
```

### Step 3: Data Preparation
```
1. Normalize data
2. Reshape to BC4D4 input format
3. Create train/test splits
4. Prepare per-finger targets
```

### Step 4: Model Implementation
```
1. Implement BC4D4 class
2. Test with Tanh activation
3. Test with Softsign activation
4. Verify parameter counts match paper
```

### Step 5: Training
```
1. Train 5 models (one per finger) for each subject
2. Use Softsign activation (best results)
3. Monitor training loss
4. Save best checkpoints
```

### Step 6: Evaluation
```
1. Calculate Pearson correlation per finger
2. Calculate average per subject
3. Calculate overall average
4. Compare with paper benchmarks
```

### Step 7: Comparison
```
1. Run FingerFlex on same data
2. Compare BC4D4 vs FingerFlex results
3. Document improvements
```

---

## Notes & Observations

### Why BC4D4 Works Better:
1. **Outlier removal** is critical - FingerFlex doesn't do this
2. **Activation function** matters - Tanh/Softsign match data distribution
3. **Simpler architecture** - No need for U-Net complexity
4. **No pooling** - Preserves information for small datasets
5. **Per-finger models** - Better specialization

### Potential Improvements Beyond Paper:
- Ensemble of Tanh and Softsign models
- Cross-validation for contamination tuning
- Hyperparameter search
- Combine with FingerFlex's wavelet features

---

## Timeline & Progress Tracking

- [ ] **Phase 1:** Data Understanding (Current)
- [ ] **Phase 2:** Isolation Forest Preprocessing
- [ ] **Phase 3:** Model Architecture
- [ ] **Phase 4:** Training Configuration
- [ ] **Phase 5:** Evaluation
- [ ] **Phase 6:** Implementation Complete
- [ ] **Phase 7:** Results Verification

---

## References

1. Jangir et al. (2025) - BC4D4 Paper
2. Lomtev et al. (2022) - FingerFlex: arXiv:2211.01960
3. BCI Competition IV Dataset 4 - University of Washington
