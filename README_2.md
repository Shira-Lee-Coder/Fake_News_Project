# Deep Learning Project - Fake News Classification

## Hyperparameter Experiments & Visualization

This module handles hyperparameter experiments and training result visualizations.
All figures are saved automatically to the `figures/` folder after running.

---

### ▶️ Run

```bash
python performance_comparison.py
```

---

### 🔬 Experiments Overview

#### Baseline Training Curve (Fig 1 & 2)
Single run with default hyperparameters for each loss function.

```python
# Fig 1 — CrossEntropyLoss
# Fig 2 — FocalLoss(γ=2.0)
# Fixed: LR = 0.001, Batch Size = 32
```

#### Learning Rate Comparison (Fig 3 & 4)
Trains the model across 4 different learning rates and compares results side by side.

```python
lr_list = [0.1, 0.01, 0.001, 0.0001]
# Fixed: Batch Size = 32
```

#### Batch Size Comparison (Fig 5 & 6)
Trains the model across 5 different batch sizes and compares results side by side.

```python
bs_list = [8, 16, 32, 64, 128]
# Fixed: LR = 0.001
```

> All experiments related to learning rate and batch size are each run **twice** — once with `CrossEntropyLoss`, once with `FocalLoss(γ=2.0)`.


---

### 📈 Output Figures

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

### 🗂️ Related Files

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
