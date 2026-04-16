import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from data_loader import load_and_preprocess_data

# Configuration
OUTPUT_FOLDER = "results_plots"
JSON_FILE = "model_results.json"
OUTPUT_FILENAME = "pytorch_mlp_predictions.png"
FIG_SIZE = (14, 10)

# Define the PyTorch MLP model structure
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_pytorch_mlp(X_train_tensor, y_train_tensor, input_dim, num_classes):
    model = MLP(input_dim=input_dim, hidden_dim=128, output_dim=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Convert to DataLoader
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    model.eval()
    return model

def generate_detailed_visualization():
    print("Start generating detailed visualizations of the first 100 pieces of data...")
    
    # 1. Check and create the output folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # 2. Read the best model name
    if not os.path.exists(JSON_FILE):
        print(f"❌ 错误：找不到 {JSON_FILE}。请先运行 model_comparison.py 生成结果。")
        return
    
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    model_names = list(results.keys())
    accuracies = [metrics["accuracy"] for metrics in results.values()]
    times = [metrics["time"] for metrics in results.values()]
    
    fig, ax1 = plt.subplots(figsize=(12, 7))

    colors = ['#3498db' if x < max(accuracies) else '#e74c3c' for x in accuracies]
    bars = ax1.bar(model_names, accuracies, color=colors, alpha=0.8, label='Accuracy')
    
    ax1.set_xlabel('Model Names', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (Detailed)', fontsize=12, color='#2c3e50')
    ax1.set_ylim(0.75, 0.8)
    ax1.tick_params(axis='y', labelcolor='#2c3e50')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    ax2 = ax1.twinx() 
    ax2.plot(model_names, times, color='#2ecc71', marker='o', linewidth=2, label='Time (s)')
    ax2.set_ylabel('Time (Seconds)', fontsize=12, color='#27ae60')
    ax2.tick_params(axis='y', labelcolor='#27ae60')

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', 
                 ha='center', va='bottom', fontweight='bold')

    plt.title("Model Performance: Accuracy vs Training Time", fontsize=16, pad=20)
    ax1.set_xticklabels(model_names, rotation=25)
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    comparison_path = os.path.join(OUTPUT_FOLDER, "model_performance_comparison.png")
    plt.savefig(comparison_path, dpi=200)
    print(f"✅ The image has been saved to: {comparison_path}")
    plt.close() 
    
    best_name = max(results, key=lambda x: results[x]['accuracy'])
    print(f"Best model: {best_name} (Accuracy: {results[best_name]['accuracy']:.4f})")
    
    # 3. Reload data
    try:
        X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data()
        df = pd.read_csv("fakenews.csv").dropna() 
        texts = df['text'].values
        labels = df['label'].values 
        
        # Re-divide the test set
        from sklearn.model_selection import train_test_split
        texts_train, texts_test, y_train_raw, y_test_raw = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Take the first 100 test data
        texts_sample = texts_test[:100]
        y_true_sample = y_test_raw[:100]
        
    except Exception as e:
        print(f"Data loading error: {e}")
        return
    
    # 4. Make predictions based on the best model name
    y_pred_sample = None
    
    if "PyTorch MLP" in best_name:
        print("Retraining the PyTorch MLP model for prediction...")
        
        # Convert text to vectors
        X_train_vec = vectorizer.transform(texts_train).toarray()
        X_test_vec = vectorizer.transform(texts_test).toarray()
        
        # Convert to PyTorch tensor
        X_train_tensor = torch.tensor(X_train_vec, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_raw, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test_vec, dtype=torch.float32)
        
        # Get the input dimension and the number of categories
        input_dim = X_train_vec.shape[1]
        num_classes = len(np.unique(labels))
        
        model = train_pytorch_mlp(X_train_tensor, y_train_tensor, input_dim, num_classes)
        
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            y_pred_sample = predicted.numpy()[:100]
            
    else:
        print("please add the corresponding logic")
        return
    
    # 5. Create visual table data
    results_df = pd.DataFrame({
        "Index": range(1, 101),
        "Input (Text Preview)": [t[:50] + "..." for t in texts_sample],
        "Actual Label": y_true_sample,
        "Predicted Label": y_pred_sample
    })
    
    # 6. Drawing
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.axis('tight')
    ax.axis('off')

    col_widths = [0.05, 0.5, 0.15, 0.15]
    
    table = ax.table(
        cellText=results_df.values,
        colLabels=results_df.columns,
        cellLoc='center',
        loc='center',
        colWidths=col_widths
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.5)
    
    # Set color
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#4ECDC4')
            cell.set_text_props(weight='bold', color='white')
        else:
            if key[1] == 3:
                actual_idx = key[0] - 1
                if actual_idx < len(y_true_sample) and actual_idx < len(y_pred_sample):
                    if y_true_sample[actual_idx] == y_pred_sample[actual_idx]:
                        cell.set_facecolor('#C8E6C9')  # Correct: Light green
                    else:
                        cell.set_facecolor('#FFCDD2')  # Error: Light red  
    
    # Save the picture
    output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ The image has been saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    generate_detailed_visualization()