# Fake News Detection Model Comparison

---

## 1. Project structure

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

## 2. Environment configuration

### 2.1 Create a virtual environment (recommended)
```bash
python -m venv venv
.venv\Scripts\activate
```

### 2.2 Install dependency packages
```bash
pip install -r requirements.txt
```

---

## 3. Program running

### 3.1 Model training and comparison (time-consuming)

```bash
python model_comparison.py
```

The script will automatically:

1. Load and preprocess the fakenews.csv dataset
2. Train and evaluate 5 models in sequence (Logistic Regression, Naive Bayes, Random Forest, PyTorch MLP, DistilBERT)
3. Record the accuracy and time consumption of each model
4. Save the results to model_results.json
5. Output the final performance ranking in the console

### 3.2 Visualization of the comparison table between the prediction results of the best model and the actual labels on the first 100 test data

```bash
python visualize_results.py
```

The script will automatically:

Generate a comparison table of the prediction results of the best model on the first 100 test data with the actual labels and save the image to the results_plots folder.
