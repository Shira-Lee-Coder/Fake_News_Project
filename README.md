# Fake News Detection — Model Comparison & Hyperparameter Experiments

This project train and evaluate five models for fake news classification, selects the best-performing model (PyTorch MLP), and conducts in-depth hyperparameter experiments with full visualization support.

---

## Part 1: Model Comparison

### 1. Project structure

- `fakenews.csv`           
  Original dataset

- `config.py`              
  Configuration files (paths, hyperparameters, etc.)

- `requirements.txt`       
  List of dependent packages

- `data_loader.py`        
  Data Loading and Preprocessing Module

- `model_comparison.py`   
  Training and comparison of multiple models

- `visualize_results.py`  
  Result visualization

- `README.md`              
  Project Description Document

- `model_results.json`     
  [Automatically Generated] Comparison Results of Multiple Models

- `results/`               
  [Auto-generated] Transformer model checkpoint folder

- `results_plots/`
  [Automatically Generated] Comparison Table of Prediction Results and Actual Labels of the Best Model on the First 100 Test Data
  
---

### 2. Environment configuration

**2.1 Create a virtual environment (recommended)**
```bash
python -m venv venv
.venv\Scripts\activate
```

**2.2 Install dependency packages**
```bash
pip install -r requirements.txt
```

---

### 3. Program running

**3.1 Model training and comparison (time-consuming)**

```bash
python model_comparison.py
```

The script will automatically:

1. Load and preprocess the fakenews.csv dataset
2. Train and evaluate 5 models in sequence (Logistic Regression, Naive Bayes, Random Forest, PyTorch MLP, DistilBERT)
3. Record the accuracy and time consumption of each model
4. Save the results to model_results.json
5. Output the final performance ranking in the console

**3.2 Visualization of the comparison table between the prediction results of the best model and the actual labels on the first 100 test data**

```bash
python visualize_results.py
```

The script will automatically:

Generate a comparison table of the prediction results of the best model on the first 100 test data with the actual labels and save the image to the results_plots folder.

---

## Part 2: Hyperparameter Experiments & Visualization

PyTorch MLP was selected as the best-performing model from Part 1. This module conducts systematic hyperparameter experiments on it and visualizes the training results. All figures are saved automatically to the `figures/` folder after running.

---

### ▶️ Run

```bash
python performance_comparison.py
```

---

### Experiments Overview

**Baseline Training Curve (Fig 1 & 2):**
Single run with default hyperparameters for each loss function.

```python
# Fig 1 — CrossEntropyLoss
# Fig 2 — FocalLoss(γ=2.0)
# Fixed: LR = 0.001, Batch Size = 32
```

**Learning Rate Comparison (Fig 3 & 4):**
Trains the model across 4 different learning rates and compares results side by side.

```python
lr_list = [0.1, 0.01, 0.001, 0.0001]
# Fixed: Batch Size = 32
```

**Batch Size Comparison (Fig 5 & 6):**
Trains the model across 5 different batch sizes and compares results side by side.

```python
bs_list = [8, 16, 32, 64, 128]
# Fixed: LR = 0.001
```

> All experiments related to learning rate and batch size are each run **twice** — once with `CrossEntropyLoss`, once with `FocalLoss(γ=2.0)`.


---

### Output Figures

| Figure | Experiment | Loss Function |
|---|---|---|
| `fig1.png` | Baseline Training Curve | CrossEntropyLoss |
| `fig2.png` | Baseline Training Curve | FocalLoss (γ=2.0) |
| `fig3.png` | Learning Rate Comparison | CrossEntropyLoss |
| `fig4.png` | Learning Rate Comparison | FocalLoss (γ=2.0) |
| `fig5.png` | Batch Size Comparison | CrossEntropyLoss |
| `fig6.png` | Batch Size Comparison | FocalLoss (γ=2.0) |

**Fig 1 & 2** — single training curve showing:
- **Training Loss** (left axis)
- **Training Accuracy & Test Accuracy** (right axis), with ★ marking the best epoch

**Fig 3 & 4** — each contains **3 subplots** comparing across learning rates:
- Training Loss / Training Accuracy / Test Accuracy

**Fig 5 & 6** — each contains **3 subplots** comparing across batch sizes:
- Training Loss / Training Accuracy / Test Accuracy

---

### Related Files

```
├── performance_comparison.py    # Entry point for all experiments
├── figures/
│   ├── fig1.png
│   ├── fig2.png
│   ├── fig3.png
│   ├── fig4.png
│   ├── fig5.png
│   └── fig6.png
```

