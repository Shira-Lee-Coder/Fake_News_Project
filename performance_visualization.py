import os, torch, warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from data_loader import load_and_preprocess_data

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────
# Network Architecture
# ─────────────────────────────────────────
class SimpleDeepNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)

# ─────────────────────────────────────────
# Custom Loss Function
# ─────────────────────────────────────────
class FocalLoss(nn.Module):
    """Reduces loss weight on easy examples so the model focuses on hard ones."""
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce  = F.cross_entropy(inputs, targets, reduction='none')
        pt  = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ─────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────
def train_pytorch_model(model, X_train, y_train, X_test, y_test,
                        epochs=10, batch_size=32, lr=0.001, criterion=None):
    le          = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train.toarray(), y_train_enc, test_size=0.1, random_state=42
    )

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
        batch_size=batch_size, shuffle=True
    )
    X_test_t = torch.FloatTensor(X_test.toarray())

    criterion = criterion or nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history                   = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    best_test_acc, best_epoch = 0.0, 1

    for epoch in range(epochs):
        model.train()
        total_loss, all_preds, all_labels = 0, [], []

        for bx, by in train_loader:
            optimizer.zero_grad()
            out  = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(torch.max(out, 1)[1].numpy())
            all_labels.extend(by.numpy())

        train_loss = total_loss / len(train_loader)
        train_acc  = accuracy_score(all_labels, all_preds)

        model.eval()
        with torch.no_grad():
            test_acc = accuracy_score(
                y_test_enc,
                torch.max(model(X_test_t), 1)[1].numpy()
            )

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc, best_epoch = test_acc, epoch + 1

        print(f"  Epoch {epoch+1:>3}/{epochs}  "
              f"Loss: {train_loss:.4f}  "
              f"Train Acc: {train_acc:.4f}  "
              f"Test Acc: {test_acc:.4f}")

    return best_test_acc, history, best_epoch


# ─────────────────────────────────────────
# Plot: Training Curve
# ─────────────────────────────────────────
def plot_training_curve(history, best_ep, best_acc,
                        lr, batch_size, loss_name="CrossEntropyLoss",
                        save_path="figures/fig.png"):
    epochs_x     = range(1, len(history['train_loss']) + 1)
    best_acc_val = history['test_acc'][best_ep - 1]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#f9f9f9')
    ax1.set_facecolor('#f9f9f9')

    # Loss on left axis
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12, color='tab:red')
    l1, = ax1.plot(epochs_x, history['train_loss'],
                   color='#e74c3c', linewidth=2, marker='o', markersize=3,
                   label='Training Loss')
    ax1.tick_params(axis='y', labelcolor='#e74c3c', labelsize=10)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.set_xlim(0.5, len(epochs_x) + 0.5)

    # Accuracy on right axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', fontsize=12, color='#2c3e50')
    l2, = ax2.plot(epochs_x, history['train_acc'],
                   color='#2980b9', linewidth=2, linestyle='--',
                   marker='s', markersize=3, label='Training Accuracy')
    l3, = ax2.plot(epochs_x, history['test_acc'],
                   color='#27ae60', linewidth=2, linestyle='--',
                   marker='^', markersize=3, label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor='#2c3e50', labelsize=10)
    ax2.set_ylim(0, 1.08)

    # Mark the best epoch with a vertical line and star
    l4 = ax1.axvline(x=best_ep, color='#e67e22', linestyle='--',
                     linewidth=2, label=f'Best Epoch ({best_ep})')
    ax2.plot(best_ep, best_acc_val, marker='*', color='#e67e22',
             markersize=18, zorder=5)
    ax2.annotate(f'Best: {best_acc_val:.4f}',
                 xy=(best_ep, best_acc_val),
                 xytext=(best_ep + 1.2, best_acc_val - 0.04),
                 fontsize=10, color='#e67e22', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#e67e22', lw=1.5))

    ax1.legend(handles=[l1, l4, l2, l3],
               labels=['Training Loss', f'Best Epoch ({best_ep})',
                       'Training Accuracy', 'Test Accuracy'],
               loc='center right', fontsize=10,
               framealpha=0.9, edgecolor='#cccccc')
    ax1.yaxis.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
    ax1.set_axisbelow(True)

    plt.title(f'Model Training Curve\n'
              f'lr={lr}  |  Batch Size={batch_size}  |  Loss={loss_name}  |  '
              f'Best Epoch={best_ep}  |  Best Test Acc={best_acc:.4f}',
              fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────
# Plot: Multi-param Comparison (3 subplots)
# ─────────────────────────────────────────
def plot_comparison(histories, best_eps, best_accs, param_list,
                    loss_name="CrossEntropyLoss", label_prefix="lr",
                    extra_info=None, save_path="figures/fig.png"):

    colors    = ['#e74c3c', '#2980b9', '#27ae60', '#8e44ad', '#f39c12']
    subtitles = ['Training Loss', 'Training Accuracy', 'Test Accuracy']
    keys      = ['train_loss',    'train_acc',         'test_acc']
    info_str  = f"  |  {extra_info}" if extra_info else ""
    best_idx  = int(np.argmax(best_accs))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#f9f9f9')
    fig.suptitle(
        f'{label_prefix.upper()} Comparison  |  Loss={loss_name}{info_str}\n'
        f'★  Best {label_prefix}={param_list[best_idx]}  '
        f'|  Best Test Acc={best_accs[best_idx]:.4f}',
        fontsize=13, fontweight='bold', y=1.04
    )

    for ax, key, subtitle in zip(axes, keys, subtitles):
        ax.set_facecolor('#f9f9f9')
        ax.set_title(subtitle, fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_xlim(0.5, len(histories[0]['train_loss']) + 0.5)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
        ax.set_axisbelow(True)

        for i, (hist, param) in enumerate(zip(histories, param_list)):
            ax.plot(range(1, len(hist[key]) + 1), hist[key],
                    color=colors[i], linewidth=2, marker='o', markersize=3,
                    label=f'{label_prefix}={param}')

        if key == 'test_acc':
            # Mark each run's best epoch
            for i, hist in enumerate(histories):
                bep  = best_eps[i]
                bval = hist['test_acc'][bep - 1]
                ax.axvline(x=bep, color=colors[i], linestyle='--',
                           linewidth=1.0, alpha=0.4, zorder=1)
                ax.plot(bep, bval, marker='*', color=colors[i],
                        markersize=13, zorder=6)

            # Build legend sorted by param value, smallest to largest
            sorted_ann = sorted(
                [(i, best_eps[i], histories[i]['test_acc'][best_eps[i]-1], colors[i])
                 for i in range(len(histories))],
                key=lambda x: param_list[x[0]]
            )
            handles = [Patch(facecolor='none', edgecolor='none',
                             label='★ = Best Accuracy')] + [
                Line2D([0], [0], color=c, linewidth=2, marker='o', markersize=4,
                       label=f'{label_prefix}={param_list[oi]}  (acc={bv:.4f})')
                for oi, _, bv, c in sorted_ann
            ]
            legend = ax.legend(handles=handles, fontsize=9, framealpha=0.93,
                               edgecolor='#aaaaaa', loc='center right',
                               handletextpad=0.4, borderpad=0.7, labelspacing=0.45)
            for t in legend.get_texts():
                if t.get_text().startswith('★'):
                    t.set_fontweight('bold')
        else:
            ax.legend(fontsize=9, framealpha=0.9, edgecolor='#cccccc')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────
# Plot: Predictions Table
# ─────────────────────────────────────────
def plot_predictions_table(model, X_test, y_test,
                           raw_texts=None, n=100,
                           save_path="figures/fig7.png"):
    X_dense = X_test.toarray()[:n]
    le      = LabelEncoder()
    y_enc   = le.fit_transform(np.array(y_test)[:n])

    model.eval()
    with torch.no_grad():
        preds = torch.max(model(torch.FloatTensor(X_dense)), 1)[1].numpy()

    # Cut long text so it fits inside the table cell
    texts = (
        [str(t)[:120] + '...' if len(str(t)) > 120 else str(t)
         for t in list(raw_texts)[:n]]
        if raw_texts is not None
        else [f"sample_{i+1}" for i in range(n)]
    )

    df = pd.DataFrame({
        'Index':        range(1, n + 1),
        'Input Text':   texts,
        'Actual Label': y_enc,
        'Pred Label':   preds,
        'Correct':      ['✓' if a == p else '✗' for a, p in zip(y_enc, preds)]
    })

    fig, ax = plt.subplots(figsize=(16, max(6, n * 0.28 + 2.0)))
    fig.patch.set_facecolor('#f9f9f9')
    ax.set_facecolor('#f9f9f9')
    ax.axis('off')

    col_labels = ['Index', 'Input Text', 'Actual Label', 'Pred Label', 'Correct']
    col_widths = [0.04, 0.5, 0.17, 0.17, 0.10]
    table_data = df[col_labels].values.tolist()

    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   colWidths=col_widths, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.25)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if row == 0:
            # Header row styling
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        elif table_data[row - 1][4] == '✗':
            # Red tint for wrong predictions
            cell.set_facecolor('#fdecea')
        else:
            # Alternating row shading for readability
            cell.set_facecolor('#ffffff' if row % 2 == 0 else '#f4f6f8')
        if col == 1 and row > 0:
            cell.set_text_props(ha='left')

    correct_n = sum(a == p for a, p in zip(y_enc, preds))
    ax.set_title(
        f'Predictions vs Actual Labels  (first {n} shown)\n'
        f'Accuracy = {correct_n/n:.4f}  ({correct_n}/{n} correct)',
        fontsize=11, fontweight='bold', pad=14
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────
# Main Function
# ─────────────────────────────────────────
def main():
    os.makedirs("figures", exist_ok=True)

    data = load_and_preprocess_data()
    if data is None:
        print("Data loading failed.")
        return

    X_train, X_test, y_train, y_test, _ = data
    input_dim = X_train.shape[1]
    LR        = 0.001
    BS        = 32

    # Fig 1 & 2: one baseline run per loss function
    for fig_id, criterion, loss_name in [
        (1, nn.CrossEntropyLoss(),  "CrossEntropyLoss"),
        (2, FocalLoss(gamma=2.0),   "FocalLoss(gamma=2.0)"),
    ]:
        print(f"\nFig {fig_id} Training...")
        acc, hist, ep = train_pytorch_model(
            SimpleDeepNet(input_dim), X_train, y_train, X_test, y_test,
            epochs=10, batch_size=BS, lr=LR, criterion=criterion
        )
        plot_training_curve(hist, ep, acc, LR, BS,
                            loss_name=loss_name,
                            save_path=f"figures/fig{fig_id}.png")

    # Fig 3 & 4: try each learning rate and compare results
    lr_list = [0.1, 0.01, 0.001, 0.0001]
    for fig_id, criterion, loss_name in [
        (3, nn.CrossEntropyLoss(),  "CrossEntropyLoss"),
        (4, FocalLoss(gamma=2.0),   "FocalLoss(gamma=2.0)"),
    ]:
        print(f"\n{loss_name} — Learning Rate Comparison...")
        results = [
            train_pytorch_model(
                SimpleDeepNet(input_dim), X_train, y_train, X_test, y_test,
                epochs=10, batch_size=BS, lr=lr, criterion=criterion
            ) for lr in lr_list
        ]
        accs, hists, eps = zip(*results)
        plot_comparison(list(hists), list(eps), list(accs), lr_list,
                        loss_name=loss_name, label_prefix="lr",
                        extra_info=f"Batch Size={BS}",
                        save_path=f"figures/fig{fig_id}.png")

    # Fig 5 & 6: try each batch size and compare results
    bs_list = [8, 16, 32, 64, 128]
    for fig_id, criterion, loss_name in [
        (5, nn.CrossEntropyLoss(),  "CrossEntropyLoss"),
        (6, FocalLoss(gamma=2.0),   "FocalLoss(gamma=2.0)"),
    ]:
        print(f"\n{loss_name} — Batch Size Comparison...")
        results = [
            train_pytorch_model(
                SimpleDeepNet(input_dim), X_train, y_train, X_test, y_test,
                epochs=10, batch_size=bs, lr=LR, criterion=criterion
            ) for bs in bs_list
        ]
        accs, hists, eps = zip(*results)
        plot_comparison(list(hists), list(eps), list(accs), bs_list,
                        loss_name=loss_name, label_prefix="bs",
                        extra_info=f"LR={LR}",
                        save_path=f"figures/fig{fig_id}.png")

if __name__ == "__main__":
    main()
