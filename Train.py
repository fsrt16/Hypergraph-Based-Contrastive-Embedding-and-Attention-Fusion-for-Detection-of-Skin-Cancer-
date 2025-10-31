

import os
import json
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
import umap


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class GraphTraining:
    def __init__(self, node_features, labels, edge_index, model, criterion, device, save_path, class_names, 
                 batch_size=32, num_epochs=1000, val_split=0.2, lr=0.001, weight_decay=1e-4, patience=20):
        self.node_features = node_features
        self.labels = labels
        self.edge_index = edge_index
        self.model = model
        self.criterion = criterion
        self.device = device
        self.save_path = save_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.class_names = class_names
        self.val_split = val_split
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience

        os.makedirs(self.save_path, exist_ok=True)
        self.save_dir = os.path.join(self.save_path, "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_path, "tensorboard"))

        self.loss_history = {'train': [], 'val': []}

        self.train_loader, self.val_loader = self._create_dataloaders()

        self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        self.early_stopping = EarlyStopping(patience=self.patience)

    def _create_dataloaders(self):
        dataset = Data(x=self.node_features, edge_index=self.edge_index, y=self.labels)
        num_nodes = dataset.num_nodes
        num_train = int((1 - self.val_split) * num_nodes)
        indices = torch.randperm(num_nodes)
        train_idx, val_idx = indices[:num_train], indices[num_train:]

        train_data = Data(x=dataset.x[train_idx], edge_index=self.edge_index, y=dataset.y[train_idx])
        val_data = Data(x=dataset.x[val_idx], edge_index=self.edge_index, y=dataset.y[val_idx])

        train_loader = DataLoader([train_data], batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader([val_data], batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            train_loss = self._train_one_epoch()
            val_loss = self._validate()

            self.loss_history['train'].append(train_loss)
            self.loss_history['val'].append(val_loss)

            self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)

            print(f"Epoch [{epoch}/{self.num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            torch.save(checkpoint, os.path.join(self.save_dir, f"epoch_{epoch}.pt"))

            self.scheduler.step(val_loss)
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("Early stopping triggered!")
                break

        self._save_loss_curves()
        self._save_loss_data()
        print("Training complete.")

        self.model.eval()
        with torch.no_grad():
            full_data = Data(x=self.node_features, edge_index=self.edge_index, y=self.labels).to(self.device)
            embeddings = self.model(full_data.x, full_data.edge_index)
        self.visualize_embeddings(embeddings, full_data.y, title="Final Embeddings")

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for data_batch in self.train_loader:
            data_batch = data_batch.to(self.device)
            self.optimizer.zero_grad()
            embeddings = self.model(data_batch)
            loss = self.criterion(embeddings, data_batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data_batch in self.val_loader:
                data_batch = data_batch.to(self.device)
                embeddings = self.model(data_batch.x, data_batch.edge_index)
                loss = self.criterion(embeddings, data_batch.y)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def _save_loss_curves(self):
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=np.arange(1, len(self.loss_history['train'])+1), y=self.loss_history['train'], label='Train Loss')
        sns.lineplot(x=np.arange(1, len(self.loss_history['val'])+1), y=self.loss_history['val'], label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, "loss_curve.png"), dpi=300)
        plt.show()

    def _save_loss_data(self):
        csv_path = os.path.join(self.save_path, "loss_data.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss'])
            for epoch, (train, val) in enumerate(zip(self.loss_history['train'], self.loss_history['val']), 1):
                writer.writerow([epoch, train, val])

        json_path = os.path.join(self.save_path, "loss_data.json")
        with open(json_path, 'w') as f:
            json.dump(self.loss_history, f, indent=4)

    def visualize_embeddings(self, embeddings, labels, title='Embedding Visualization'):
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        labels = labels.astype(int)
        label_names = np.array([self.class_names[label] for label in labels])

        palette = sns.color_palette("hls", len(np.unique(label_names)))

        tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42)
        embeddings_tsne = tsne.fit_transform(embeddings)

        reducer = umap.UMAP(n_components=2, random_state=42)
        embeddings_umap = reducer.fit_transform(embeddings)

        fig, axes = plt.subplots(1, 2, figsize=(22, 10))

        sns.scatterplot(
            x=embeddings_tsne[:, 0],
            y=embeddings_tsne[:, 1],
            hue=label_names,
            palette=palette,
            alpha=0.8,
            ax=axes[0],
            legend='full'
        )
        axes[0].set_title('t-SNE', fontsize=18)
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        axes[0].grid(True)

        sns.scatterplot(
            x=embeddings_umap[:, 0],
            y=embeddings_umap[:, 1],
            hue=label_names,
            palette=palette,
            alpha=0.8,
            ax=axes[1],
            legend=False
        )
        axes[1].set_title('UMAP', fontsize=18)
        axes[1].set_xlabel('UMAP 1')
        axes[1].set_ylabel('UMAP 2')
        axes[1].grid(True)

        plt.suptitle(title, fontsize=22)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, "embedding_visualization.png"), dpi=300)
        plt.show()
