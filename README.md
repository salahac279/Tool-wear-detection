# CNC Tool Wear Classifier

A machine learning project for binary classification of CNC cutting tool condition (Healthy vs. Worn) using spindle sensor data collected from a **DMG Mori US40** CNC machine with a **Siemens Sinumerik** controller.

Two approaches are implemented and compared: a **Random Forest classifier** (scikit-learn) and a **Neural Network** (PyTorch).

---

## Problem Statement

Tool wear in CNC machining degrades surface quality, increases cutting forces, and can cause catastrophic tool failure. This project explores whether spindle sensor signals can reliably distinguish between a healthy (new) tool and a worn tool during active cutting — without interrupting the machining process.

---

## Machine & Data Setup

| Parameter | Value |
|---|---|
| Machine | DMG Mori US40 |
| Controller | Siemens Sinumerik |
| Sampling rate | 10 samples/sec (100ms intervals) |
| Cutting condition filter | `actFeedRate == 1852` (steady-state only) |
| Data acquisition | OPC-UA → Node-RED → InfluxDB v2 |

### Recorded Channels

| Column | Description |
|---|---|
| `aaTorque` | Spindle torque |
| `aaLoad` | Spindle load |
| `actSpeedRel` | Relative spindle speed |
| `actFeedRate` | Actual feed rate |
| `aaPower` | Spindle power (available, not used yet) |

### Data Slicing

Raw CSVs contain multiple experiments. The relevant 10-minute cutting blocks are extracted manually by row index:

| File | Rows Used |
|---|---|
| `new1.csv` (Healthy tool) | `iloc[3374:7280]` |
| `old1.csv` (Worn tool) | `iloc[890:4470]` |

---

## Feature Engineering

Raw time-series data is segmented into **5-second windows** (50 samples at 10Hz). Each window is summarised into 5 statistical features:

| Feature | Description |
|---|---|
| `Torque_Mean` | Mean spindle torque over the window |
| `Torque_Std` | Torque variability (standard deviation) |
| `Load_Mean` | Mean spindle load |
| `Load_Std` | Load variability |
| `SpeedRel_Std` | Relative speed variability — strongest discriminating feature |

### Key Finding: Air-Cut Filtering

Filtering on `actFeedRate == 1852` (exact steady-state feed) rather than `> 500` was critical. The broader filter included ramp-up/ramp-down transients which corrupted the feature distributions and made the two classes nearly indistinguishable. Switching to the exact feed rate pushed training accuracy from ~72% to ~90%.

---

## Models

### 1. Random Forest (`sklearn_rf.py`)
- 100 decision trees, `random_state=42`
- 70/30 train/test split
- Outputs: accuracy, classification report, confusion matrix, feature importances, per-window prediction probabilities

### 2. Neural Network (`pytorch_nn.py`)
- Architecture: `Linear(5→32) → ReLU → Linear(32→16) → ReLU → Linear(16→8) → ReLU → Linear(8→4) → ReLU → Linear(4→1)`
- Loss: `BCEWithLogitsLoss`
- Optimizer: Adam, lr=0.0008
- 5000 epochs, 70/30 train/test split, `random_state=42`
- Device-agnostic (runs on CPU or CUDA GPU)
- Full seed control for reproducibility (see Reproducibility section)

> **Known limitations in current version:** feature normalization (StandardScaler) and early stopping are not yet implemented. Overfitting begins around epoch 1000–1200. These are planned improvements.

---

## Results

| Model | Train Accuracy | Test Accuracy | Notes |
|---|---|---|---|
| Random Forest | — | ~80–88% (CV) | 5-fold cross-validation, `actFeedRate == 1852` |
| Neural Network (Adam) | ~90% | ~71% (epoch 2000) | Best generalization before overfitting sets in |

> **On the neural network result:** the model reaches its best test loss (~0.59) around epoch 2000, which is the honest result to report. Training continues to improve beyond this point — reaching 97% train accuracy and 78% test accuracy by epoch 4900 — but test loss simultaneously climbs to 1.26, indicating the model is memorizing training samples rather than generalizing. The late 78% figure is not a reliable result. Early stopping, Dropout, and feature normalization are planned improvements to address this.

> **On the filter change:** switching from `actFeedRate > 500` to `== 1852` was the single most impactful change in the project, pushing training accuracy from ~72% to ~90% by isolating steady-state cutting and removing ramp-up/ramp-down transients from the data.

### Feature Importance (Random Forest)

`SpeedRel_Std` is the strongest discriminating feature (~35% importance), suggesting worn tools introduce measurable spindle speed variability during cutting. The remaining four features contribute roughly equally at 15–18% each.

---

## Project Structure

```
cnc-tool-wear-classifier/
│
├── sklearn_rf.py          # Random Forest classifier (full pipeline)
├── pytorch_nn.py          # Neural network classifier (full pipeline)
│
├── data/
│   └── README.md          # Data format description (CSVs not included)
│
├── outputs/
│   ├── confusion_matrix_final.png
│   └── feature_importance.png
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### Requirements

```bash
pip install -r requirements.txt
```

```
torch
pandas
numpy
scikit-learn
matplotlib
seaborn
scipy
```

### Running

Update the CSV paths at the top of each script to point to your local data files:

```python
df_new = pd.read_csv("path/to/new1.csv")
df_old = pd.read_csv("path/to/old1.csv")
```

Then run either script:

```bash
python sklearn_rf.py
python pytorch_nn.py
```

---

## Reproducibility

All random sources are seeded for reproducibility:

```python
import random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## Data

Raw CSV files are **not included** in this repository (industrial machine data). The expected format is a flat CSV with at minimum the columns listed in the Data Setup section above, sampled at 10Hz during CNC milling operations.

---

## Status

- [x] Data pipeline (OPC-UA → InfluxDB)
- [x] Feature extraction with sliding windows
- [x] Air-cut filtering (steady-state isolation via `actFeedRate == 1852`)
- [x] Random Forest baseline with confusion matrix and feature importance
- [x] Neural network (Adam optimizer, wider architecture)
- [x] Full seed control for reproducibility
- [x] Feature analysis (PCA, t-SNE, correlation, window size sensitivity)
- [ ] Feature normalization (StandardScaler)
- [ ] Early stopping + best model saving
- [ ] Dropout regularization
- [ ] Frequency-domain features (FFT)
- [ ] Larger dataset collection
- [ ] Real-time inference pipeline
