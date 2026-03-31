import config, json, torch, gc, sys, os, time, warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from data_loader import load_and_preprocess_data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore")

## --- MLP --- ##
class SimpleDeepNet(nn.Module):
    def __init__(self, input_dim):
        super(SimpleDeepNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2) 
        )

    def forward(self, x):
        return self.network(x)

def train_pytorch_model(model, X_train, y_train, X_test, y_test, epochs=5, batch_size=32):
    # Data conversion
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()
    
    # Label encoding
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    # Convert to PyTorch Tensor
    X_train_t = torch.FloatTensor(X_train_dense)
    y_train_t = torch.LongTensor(y_train_enc)
    X_test_t = torch.FloatTensor(X_test_dense)
    y_test_t = torch.LongTensor(y_test_enc)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs, 1)
        acc = accuracy_score(y_test_enc, predicted.numpy())
    return acc

## --- Transformer --- ##
def train_transformer_model(texts_train, texts_test, y_train, y_test, model_name="distilbert-base-uncased", epochs=1):   
    from sklearn.metrics import accuracy_score
    import numpy as np
    
    # 1. Load the tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # 2. Data encoding
    train_encodings = tokenizer(texts_train.astype(str).tolist(), truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(texts_test.astype(str).tolist(), truncation=True, padding=True, max_length=128)
    
    # 3. Tag Encoding (Unified Mapping)
    all_labels = pd.concat([y_train, y_test])
    le = LabelEncoder()
    le.fit(all_labels)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    
    # 4. Custom Dataset
    class NewsDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)
            
    train_dataset = NewsDataset(train_encodings, y_train_enc)
    test_dataset = NewsDataset(test_encodings, y_test_enc)
    
    # 5. Define evaluation metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, predictions)}
    
    # 6. Training parameters
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="no",
        disable_tqdm=True, 
        report_to="none" 
    )
    
    # 7. Create a Trainer (pass in compute_metrics)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    # 8. Evaluation
    print("Evaluating...")
    results = trainer.evaluate()
    acc = results.get('eval_accuracy', 0.0)
    
    # Clean up memory
    del model, trainer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return acc

## --- Main Program: Model Comparison --- ##
def run_model_comparison():
    print("="*40)
    print("Start model comparison")
    print("="*40)

    # 1. Obtain data
    df = pd.read_csv(config.DATA_FILE).dropna(subset=[config.TEXT_COLUMN, config.LABEL_COLUMN])
    all_texts = df[config.TEXT_COLUMN].astype(str)
    all_labels = df[config.LABEL_COLUMN]

    # Divide the dataset
    from sklearn.model_selection import train_test_split

    texts_train, texts_test, y_train, y_test = train_test_split(
        all_texts, all_labels, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # Get TF-IDF data
    data = load_and_preprocess_data()
    if data is None: 
        return
    
    X_train, X_test, y_train, y_test, _ = data
    input_dim = X_train.shape[1]

    input_dim = X_train.shape[1]

    # 2. Define 5 model configurations
    models_config = [
        {
            "name": "1. Logistic Regression",
            "type": "sklearn",
            "model": LogisticRegression(max_iter=1000, random_state=42)
        },
        {
            "name": "2. Naive Bayes",
            "type": "sklearn",
            "model": MultinomialNB()
        },
        {
            "name": "3. Random Forest",
            "type": "sklearn",
            "model": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1) 
        },
        {
            "name": "4. PyTorch MLP",
            "type": "pytorch_mlp",
            "model": SimpleDeepNet(input_dim)
        },
        {
            "name": "5. Transformer (DistilBERT)",
            "type": "transformer",
            "model": "distilbert-base-uncased"
        }
    ]

    results = {}

    # 3. Loop training
    for config_item in models_config:
        name = config_item["name"]
        m_type = config_item["type"]
        model = config_item["model"]
        
        print(f"\n Processing: {name} ...")
        start_time = time.time()
        
        try:
            if m_type == "sklearn":
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                
            elif m_type == "pytorch_mlp":
                acc = train_pytorch_model(model, X_train, y_train, X_test, y_test, epochs=5, batch_size=32)
                
            elif m_type == "transformer":
                acc = train_transformer_model(texts_train, texts_test, y_train, y_test, epochs=1)
            
            end_time = time.time()
            duration = end_time - start_time
            
            results[name] = {"accuracy": acc, "time": duration}
            
            print(f"✅ [{name}] Completed!")
            print(f"   Accuracy: {acc:.4f}")
            print(f"   Time: {duration:.2f} s")
            
        except Exception as e:
            print(f"❌ [{name}] Error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"accuracy": 0.0, "time": 0.0, "error": str(e)}

    # 4. Summary Report
    print("\n" + "="*40)
    print("Final comparison results")
    print("="*40)

    valid_results = {k: v for k, v in results.items() if v['accuracy'] > 0}
    
    if not results:
        print("No model ran successfully.")
        return

    # Find the best model
    best_name = max(valid_results, key=lambda x: valid_results[x]["accuracy"])
    best_acc = valid_results[best_name]["accuracy"]

    # Output in sorted order
    sorted_models = sorted(valid_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for rank, (name, res) in enumerate(sorted_models, 1):
        marker = "👑" if name == best_name else f"{rank}."
        t_flag = "(Transformer)" if "Transformer" in name else ""
        print(f"{marker} {name} {t_flag}: {res['accuracy']:.4f} (耗时 {res['time']:.1f}s)")

    print(f"\n The model that performs the best is【{best_name}.")

    return results

if __name__ == "__main__":
    results = run_model_comparison()   
    output_file = "model_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        clean_results = {}
        for name, data in results.items():
            clean_results[name] = {
            "accuracy": float(data["accuracy"]),
            "time": float(data["time"])
        }
        json.dump(clean_results, f, indent=4, ensure_ascii=False)

    print(f"\n The results have been automatically saved to:{os.path.abspath(output_file)}")