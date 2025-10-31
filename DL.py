import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------- Transformer Block ----------------
class TransformerBlock(nn.Module):
    def __init__(self, input_dim: int = 64, heads: int = 4, mlp_dim: int = 128):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        return self.norm2(x + mlp_out)

# ---------------- Transformer Backbone ----------------
class TransformerBackbone(nn.Module):
    def __init__(self, seq_len: int, dim: int, depth: int, heads: int):
        super().__init__()
        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.blocks(x))

# ---------------- ViT Encoder ----------------
class ViTEncoder2(nn.Module):
    def __init__(self, input_dim: int = 64, patch_size: int = 8, dim: int = 64, depth: int = 4, heads: int = 4, num_classes: int = 4):
        super().__init__()
        assert input_dim % patch_size == 0
        self.num_patches = input_dim // patch_size
        self.patch_embed = nn.Linear(patch_size, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.transformer = TransformerBackbone(self.num_patches + 1, dim, depth, heads)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = x.view(B, self.num_patches, -1)
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.head(x[:, 0])

# ---------------- Swin Encoder ----------------
class SwinEncoder(nn.Module):
    def __init__(self, input_dim: int = 64, window_size: int = 4, dim: int = 64, depth: int = 4, heads: int = 4, num_classes: int = 4):
        super().__init__()
        assert input_dim % window_size == 0
        self.num_windows = input_dim // window_size
        self.embed = nn.Linear(window_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_windows, dim))
        self.transformer = TransformerBackbone(self.num_windows, dim, depth, heads)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = x.view(B, self.num_windows, -1)
        x = self.embed(x) + self.pos_embed
        x = self.transformer(x)
        return self.head(x.mean(dim=1))

# ---------------- DeiT Encoder ----------------
class DeiTEncoder(nn.Module):
    def __init__(self, input_dim: int = 64, patch_size: int = 8, dim: int = 64, depth: int = 4, heads: int = 4, num_classes: int = 4):
        super().__init__()
        assert input_dim % patch_size == 0
        self.num_patches = input_dim // patch_size
        self.patch_embed = nn.Linear(patch_size, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.distill_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 2, dim))
        self.transformer = TransformerBackbone(self.num_patches + 2, dim, depth, heads)
        self.head = nn.Linear(dim, num_classes)
        self.head_dist = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = x.view(B, self.num_patches, -1)
        x = self.patch_embed(x)
        cls_tok = self.cls_token.expand(B, -1, -1)
        dist_tok = self.distill_token.expand(B, -1, -1)
        x = torch.cat([cls_tok, dist_tok, x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        cls_out = self.head(x[:, 0])
        dist_out = self.head_dist(x[:, 1])
        return (cls_out + dist_out) / 2

# ------------------------- Dataset Wrapper -------------------------
class TensorDatasetWrapper(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ------------------------- Model Architectures -------------------------
class CNN1DModel(nn.Module):
    def __init__(self, input_length=64, num_classes=4):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

class Attention1DModel(nn.Module):
    def __init__(self, input_dim=64, embed_dim=64, num_classes=4):
        super(Attention1DModel, self).__init__()
        self.query = nn.Linear(input_dim, embed_dim)
        self.key = nn.Linear(input_dim, embed_dim)
        self.value = nn.Linear(input_dim, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        attn_weights = torch.softmax(torch.bmm(Q.unsqueeze(1), K.unsqueeze(2)) / np.sqrt(Q.shape[1]), dim=-1)
        attn_output = torch.bmm(attn_weights, V.unsqueeze(1)).squeeze(1)
        x = self.fc1(attn_output)
        return self.fc2(F.relu(x))

class LayerAttention(nn.Module):
    def __init__(self, input_dim=64, num_classes=4):
        super(LayerAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, batch_first=True)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.mean(dim=1)
        return self.fc(x)

class ChannelAttention(nn.Module):
    def __init__(self, input_dim=64, num_classes=4):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, input_dim)
        self.sigmoid = nn.Sigmoid()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        avg_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        avg_out = self.fc2(F.relu(self.fc1(avg_pool)))
        max_out = self.fc2(F.relu(self.fc1(max_pool)))
        out = self.sigmoid(avg_out + max_out)
        x = x * out.unsqueeze(1)
        x = x.mean(dim=1)
        return self.classifier(x)

class LSTMModel(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, num_layers=2, num_classes=4):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim=64, num_classes=4):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class ModelTrainerEvaluator:
    def __init__(self, model, model_name, 
                 X_train, y_train, X_test, y_test, 
                 num_classes, classes, batch_size=32,
                 global_stats_df=None, classwise_df=None):

        self.model = model
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Class = classes
        # Convert inputs to tensors if needed
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.long)
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.tensor(X_test, dtype=torch.float32)
        if not isinstance(y_test, torch.Tensor):
            y_test = torch.tensor(y_test, dtype=torch.long)

        # Datasets and Loaders
        self.train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

        # Optional DataFrames
        self.global_stats_df = global_stats_df if global_stats_df is not None else pd.DataFrame()
        self.classwise_df = classwise_df if classwise_df is not None else pd.DataFrame()

    def train(self, epochs=10, lr=0.001):
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[{self.model_name}] Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    def evaluate(self):
        self.model.eval()
        y_true, y_pred, y_scores = [], [], []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()

                y_true.extend(y_batch.numpy())
                y_pred.extend(preds)
                y_scores.extend(probs.cpu().numpy())

        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        y_scores_np = np.array(y_scores)

        acc = accuracy_score(y_true_np, y_pred_np)
        cm = confusion_matrix(y_true_np, y_pred_np)
        report = classification_report(y_true_np, y_pred_np, output_dict=True)

        print(f"\nClassification Report for {self.model_name}:\n", classification_report(y_true_np, y_pred_np, target_names=self.Class))
        print("Accuracy:", acc)

        # Confusion Matrix Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=self.Class, yticklabels=self.Class)
        plt.title(f"{self.model_name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        # ROC AUC Score
        if self.num_classes > 2:
            roc_auc = roc_auc_score(y_true_np, y_scores_np, multi_class='ovr')
        else:
            roc_auc = roc_auc_score(y_true_np, y_scores_np[:, 1])
        print("ROC AUC:", roc_auc)
                # ROC Curve Plot
        fpr = dict()
        tpr = dict()
        roc_auc_dict = dict()

        # Binarize the true labels for multi-class
        y_true_bin = label_binarize(y_true_np, classes=np.arange(self.num_classes))

        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores_np[:, i])
            roc_auc_dict[i] = auc(fpr[i], tpr[i])

        # Plotting
        plt.figure(figsize=(8, 6))
        for i in range(self.num_classes):
            plt.plot(fpr[i], tpr[i], label=f'Class {self.Class[i]} (AUC = {roc_auc_dict[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'{self.model_name} - ROC Curves')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()


        # Global statistics update
        global_row = {
            "Model": self.model_name,
            "Accuracy": acc,
            "ROC_AUC": roc_auc
        }
        self.global_stats_df = pd.concat([self.global_stats_df, pd.DataFrame([global_row])], ignore_index=True)

        # Class-wise metrics update
        for class_label, metrics in report.items():
            if class_label.isdigit():
                row = {
                    "Model": self.model_name,
                    "Class": class_label,
                    "Precision": metrics['precision'],
                    "Recall": metrics['recall'],
                    "F1-Score": metrics['f1-score'],
                    "Support": metrics['support']
                }
                self.classwise_df = pd.concat([self.classwise_df, pd.DataFrame([row])], ignore_index=True)

        return self.global_stats_df, self.classwise_df
