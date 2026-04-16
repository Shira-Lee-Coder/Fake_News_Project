# Fake News Detection — Model Comparison & Hyperparameter Experiments

This project trains and evaluates five models for fake news detection, selects the best-performing model (PyTorch MLP), and conducts in-depth hyperparameter experiments with full visualization support.

---

## Part 1: Presentation Feedback

### 1. Data Cleaning & Preprocessing

This module handles the transformation of raw text data into a numerical format ready for model training. The workflow consists of the following steps:

**1.1 Data Loading:**

Uses pd.read_csv to load the CSV file, incorporating exception handling for missing files.

**1.2 Integrity Check:** 

Validates that the text and label columns specified in the configuration exist within the dataset.

**1.3 Data Cleaning:**

Uses dropna to remove rows with missing values, ensuring data quality.

**1.4 Feature Separation:** 

Extracts the text column (casting to string) and separates the features X from the labels y.

**1.5 Text Vectorization:** 

Uses TfidfVectorizer to convert text into a TF-IDF feature matrix (including stop-word removal and feature dimensionality reduction).

**1.6 Dataset Splitting:** 

Uses train_test_split to randomly partition the data into training and testing sets.
  
---

### 2. New data visualization: real news vs fake news

<img width="4860" height="2528" alt="news_wordcloud_comparison" src="https://github.com/user-attachments/assets/0bbe3915-9f9c-4ac1-bcf0-9295cfea5c6c" />

---

### 3. New data visualization: model comparision

Since the initial weights of PyTorch are randomly generated, this resulted in the two models having the same accuracy in the presentation PPT. The code has been re-run to update the results, as shown in the figure below:

<p align="center">
  <img src="https://github.com/user-attachments/assets/d0318e59-57b9-4b21-8958-45896c61f00d" width="800px">
  <br>
  <i>Figure 1: Model Running Results</i>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/886ee365-55f9-4551-b14f-b2e943b2c90f" width="800px">
  <br>
  <i>Figure 2: Comparative Analysis Chart of Accuracy and Running Time of Each Model</i>
</p>>
 
---

## Part 2: Model Comparison

### 1. Project structure

- `fakenews.csv`           
  Original dataset

- `generate_wordcloud.py`
  Word Cloud Visualization 

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

- `news_wordcloud_comparison.png`
  [Automatically Generated] Word Cloud of True and False News

- `model_results.json`     
  [Automatically Generated] Comparison Results of Multiple Models

- `model_accuracy_comparison.png`
  [Automatically Generated] Multi-model result comparison chart

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

**3.1 Word Cloud Visualization**

```bash
python generate_wordcloud.py
```

The script will automatically:

Generate a word cloud of true and false news, and save the image as news_wordcloud_comparison.png.


**3.2 Model training and comparison (time-consuming)**

```bash
python model_comparison.py
```

The script will automatically:

1. Load and preprocess the fakenews.csv dataset
2. Train and evaluate 5 models in sequence (Logistic Regression, Naive Bayes, Random Forest, PyTorch MLP, DistilBERT)
3. Record the accuracy and time consumption of each model
4. Save the results to model_results.json
5. Output the final performance ranking in the console

**3.3 Visualization of the comparison table between the prediction results of the best model and the actual labels on the first 100 test data**

```bash
python visualize_results.py
```

The script will automatically:

1. Generate a comparison table of the prediction results of the best model on the first 100 test data with the actual labels and save the image to the results_plots folder.
2. Multi-model result comparison chart.

---

## Part 3: Hyperparameter Experiments & Visualization

PyTorch MLP was selected as the best-performing model from Part 1. This module conducts systematic hyperparameter experiments on it and visualizes the training results. All figures are saved automatically to the `figures/` folder after running.

---

### ▶️ Run

```bash
python performance_visualization.py
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

