# import os
# import umap
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import torch
# import pandas as pd
# import torch
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# import torch
# from torch_geometric.data import DataLoader, Data
# import torch.nn.functional as F
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from torch.optim import Adam
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.manifold import TSNE
# import numpy as np
# import torch

import os
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import DataLoader, Data
from sklearn.manifold import TSNE
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter




class GraphTraining:
    def __init__(self, Gtrain, Gtest, GFull, model, criterion, optimizer, device, save_path,class_names, batch_size=32, num_epochs=1000):
        # Add this in the __init__ method
        
        self.best_val_loss = float('inf')
        self.logs_dir = os.path.join(save_path, "logs")
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.class_names = class_names
        self.viz = Visualizer(self.class_names)
        # Set up checkpoint directory
        self.save_dir = os.path.join(self.save_path, "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)
        self.history = {
            "train_loss": [],
            "val_loss": []
        }
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_path, 'logs'))
        # self.node_features = Gtrain.x
        # self.labels = Gtrain.y
        # self.edge_index = Gtrain.edge_index
        # # DataLoader setup
        # self.train_data = Data(x=self.node_features, 
        #                        edge_index=self.edge_index,
        #                          edge_attr=None, 
        #                          y=self.labels)
        # self.train_loader = DataLoader([self.train_data], 
        #                                batch_size=self.batch_size,
        #                                  shuffle=True)
        self.train_data = self._prepare_data(Gtrain)
        self.train_loader = DataLoader([self.train_data], 
                                       batch_size=self.batch_size,
                                       shuffle=True)
        self.test_data = self._prepare_data(Gtest)
        self.test_loader = DataLoader([self.test_data], 
                                      batch_size=self.batch_size,
                                      shuffle=False)
        self.full_data = self._prepare_data(GFull)
        self.full_loader = DataLoader([self.full_data],
                                        batch_size=self.batch_size,
                                        shuffle=False)
       
    def _prepare_data(self, G):
        node_features = G.x
        labels = G.y
        edge_index = G.edge_index
        node_features = node_features.to(self.device)
        labels = labels.to(self.device)
        edge_index = edge_index.to(self.device)
        data = Data(x=node_features, 
                               edge_index=edge_index,
                                 edge_attr=None, 
                                 y=labels)
        return data

    def train(self):
        # Training loop
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            total_loss = 0.0

            for data_batch in self.train_loader:
                data_batch = data_batch.to(self.device)
                self.optimizer.zero_grad()

                # Forward pass
                embeddings = self.model(data_batch)
                loss = self.criterion(embeddings, 
                                      data_batch.y)

                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_loader)

            # Calculate for Test set
            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for data_batch in self.test_loader:
                    data_batch = data_batch.to(self.device)
                    out = self.model(data_batch)
                    loss = self.criterion(out, data_batch.y)
                    val_loss += loss.item()
                avg_val_loss = val_loss / len(self.test_loader)

            # Track the learning rate every epoch
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning Rate', lr, epoch)

            # Log train and validation loss to TensorBoard
            self.writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

            # Save losses
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)

            print(f"Epoch [{epoch}/{self.num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")


            # Save checkpoin
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'history': self.history,
            }
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                print(f"Saving best model at epoch {epoch} with validation loss: {avg_val_loss:.4f}")

                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model.pt"))

            # Visualize embeddings every 10 epochs
            if epoch % 200 == 0:
                
                embeddingsTrain, embeddingsTest,embeddingsFull, labels_train, labels_test, labels_full = self.get_embeddings_and_labels()
                # with torch.no_grad():
                #     embeddingsTrain = self.model(self.train_data)
                #     embeddingsTest = self.model(self.test_data)
                #     embeddingsFull = self.model(self.full_data)
                self.viz.visualize_embeddings2(embeddingsTrain, embeddingsTest,embeddingsFull,
                                           labels_train, labels_test, labels_full,
                                             title=f'T-SNE after Epoch {epoch}')
                self.viz.visualize_embeddings_umap(embeddingsTrain, embeddingsTest,embeddingsFull,
                                           labels_train, labels_test, labels_full,
                                             title=f'UMAP after Epoch {epoch}')
                
        self.save_loss_curves()
        self.save_loss_history()
        # Save final model
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "final_model.pt"))
        print("Training complete. Model saved.")
        self.writer.close()



    # def visualize_embeddings(self, embeddings, labels, title='t-SNE Visualization'):
    #     # Convert to numpy if tensor
    #     if isinstance(embeddings, torch.Tensor):
    #         embeddings = embeddings.detach().cpu().numpy()
    #     if isinstance(labels, torch.Tensor):
    #         labels = labels.detach().cpu().numpy()
        
    #     # Ensure labels are integers
    #     labels = labels.astype(int)
        
    #     # Apply t-SNE
    #     tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42)
    #     embeddings_2d = tsne.fit_transform(embeddings)

    #     # Map numeric labels to class names
    #     label_names = np.array([self.class_names[label] for label in labels])

    #     # Plot
    #     plt.figure(figsize=(12, 10))
    #     palette = sns.color_palette("hls", len(np.unique(label_names)))
    #     sns.scatterplot(
    #         x=embeddings_2d[:, 0],
    #         y=embeddings_2d[:, 1],
    #         hue=label_names,
    #         palette=palette,
    #         legend='full',
    #         alpha=0.7
    #     )
    #     plt.title(title, fontsize=16)
    #     plt.xlabel('TSNE-1')
    #     plt.ylabel('TSNE-2')
    #     plt.grid(True)
    #     plt.legend(loc='best', fontsize=12, title='Classes')
    #     plt.show()

    def save_loss_curves(self):
        plt.figure(figsize=(10,6))
        sns.lineplot(x=np.arange(1, len(self.history['train_loss'])+1), y=self.history['train_loss'], label='Train Loss')
        sns.lineplot(x=np.arange(1, len(self.history['val_loss'])+1), y=self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.logs_dir, 'loss_curve.png'))
        plt.show()

    def save_loss_history(self):
        # Save as CSV
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(self.logs_dir, 'loss_history.csv'), index=False)
        # Save as JSON
        df.to_json(os.path.join(self.logs_dir, 'loss_history.json'), orient='records', lines=True)




    def getFinalEmbeddings(self):
        embeddings_train = self.get_model_embeddings(self.train_data)
        embeddings_test = self.get_model_embeddings(self.test_data)
        embeddings_full = self.get_model_embeddings(self.full_data)
        return [embeddings_train, embeddings_test, embeddings_full]


    def get_model_embeddings(self,data):
        """
        Generate embeddings for the given data using the model.
        Args:
            data: The input data for which to generate embeddings.
            
        Returns:
            embeddings: The generated embeddings.
        """
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            embeddings = self.model(data)
            return embeddings.cpu().numpy()

    def get_embeddings_and_labels(self):
        """
        Generate embeddings and labels for the train, test, and full datasets using the provided model.
        Args:
            model: The trained model used to generate embeddings (e.g., a neural network).
            train_data, test_data, full_data: The datasets for which we need the embeddings and labels.
            
        Returns:
            embeddings_train, embeddings_test, embeddings_full: The embeddings for train, test, and full datasets.
            labels_train, labels_test, labels_full: The corresponding labels for the datasets.
        """

        
        # Get embeddings for train, test, and full data
        embeddings_train = self.get_model_embeddings(self.train_data)
        embeddings_test = self.get_model_embeddings(self.test_data)
        embeddings_full = self.get_model_embeddings(self.full_data)
        
        # Get labels for train, test, and full data
        labels_train = self.train_data.y.cpu().numpy()
        labels_test = self.test_data.y.cpu().numpy()
        labels_full = self.full_data.y.cpu().numpy()

        # Ensure labels are integers (for consistency)
        labels_train = labels_train.astype(int)
        labels_test = labels_test.astype(int)
        labels_full = labels_full.astype(int)

        return embeddings_train, embeddings_test, embeddings_full, labels_train, labels_test, labels_full



    def visualize_embeddings_umap(self, embeddingTrain, embeddingTest, embeddingFull, 
                                labels_train, labels_test, labels_full, title='UMAP Visualization'):
        # Convert to numpy if tensor
        if isinstance(embeddingTrain, torch.Tensor):
            embeddingTrain = embeddingTrain.detach().cpu().numpy()
        if isinstance(embeddingTest, torch.Tensor):
            embeddingTest = embeddingTest.detach().cpu().numpy()
        if isinstance(embeddingFull, torch.Tensor):
            embeddingFull = embeddingFull.detach().cpu().numpy()

        if isinstance(labels_train, torch.Tensor):
            labels_train = labels_train.detach().cpu().numpy()
        if isinstance(labels_test, torch.Tensor):
            labels_test = labels_test.detach().cpu().numpy()
        if isinstance(labels_full, torch.Tensor):
            labels_full = labels_full.detach().cpu().numpy()

        # Ensure labels are integers
        labels_train = labels_train.astype(int)
        labels_test = labels_test.astype(int)
        labels_full = labels_full.astype(int)

        # Apply UMAP to the full, train, and test embeddings
        umap_model = umap.UMAP(n_components=2, random_state=42)
        
        # UMAP on full dataset
        embeddings_full_2d = umap_model.fit_transform(embeddingFull)
        
        # UMAP on train dataset
        embeddings_train_2d = umap_model.fit_transform(embeddingTrain)
        
        # UMAP on test dataset
        embeddings_test_2d = umap_model.fit_transform(embeddingTest)

        # Map numeric labels to class names
        label_names_train = np.array([self.class_names[label] for label in labels_train])
        label_names_test = np.array([self.class_names[label] for label in labels_test])
        label_names_full = np.array([self.class_names[label] for label in labels_full])

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Full dataset plot
        palette = sns.color_palette("hls", len(np.unique(label_names_full)))
        sns.scatterplot(
            x=embeddings_full_2d[:, 0],
            y=embeddings_full_2d[:, 1],
            hue=label_names_full,
            palette=palette,
            legend='full',
            alpha=0.7,
            ax=axes[0]
        )
        axes[0].set_title(f'{title} - Full Dataset', fontsize=14)
        axes[0].set_xlabel('UMAP-1')
        axes[0].set_ylabel('UMAP-2')
        axes[0].grid(True)

        # Training set plot
        sns.scatterplot(
            x=embeddings_train_2d[:, 0],
            y=embeddings_train_2d[:, 1],
            hue=label_names_train,
            palette=palette,
            legend='full',
            alpha=0.7,
            ax=axes[1]
        )
        axes[1].set_title(f'{title} - Training Set', fontsize=14)
        axes[1].set_xlabel('UMAP-1')
        axes[1].set_ylabel('UMAP-2')
        axes[1].grid(True)

        # Test set plot
        sns.scatterplot(
            x=embeddings_test_2d[:, 0],
            y=embeddings_test_2d[:, 1],
            hue=label_names_test,
            palette=palette,
            legend='full',
            alpha=0.7,
            ax=axes[2]
        )
        axes[2].set_title(f'{title} - Test Set', fontsize=14)
        axes[2].set_xlabel('UMAP-1')
        axes[2].set_ylabel('UMAP-2')
        axes[2].grid(True)

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()



    def visualize_embeddings2(self, embeddingTrain, embeddingTest, embeddingFull, 
                            labels_train, labels_test, labels_full, title='t-SNE Visualization'):
        # Convert to numpy if tensor
        if isinstance(embeddingTrain, torch.Tensor):
            embeddingTrain = embeddingTrain.detach().cpu().numpy()
        if isinstance(embeddingTest, torch.Tensor):
            embeddingTest = embeddingTest.detach().cpu().numpy()
        if isinstance(embeddingFull, torch.Tensor):
            embeddingFull = embeddingFull.detach().cpu().numpy()

        if isinstance(labels_train, torch.Tensor):
            labels_train = labels_train.detach().cpu().numpy()
        if isinstance(labels_test, torch.Tensor):
            labels_test = labels_test.detach().cpu().numpy()
        if isinstance(labels_full, torch.Tensor):
            labels_full = labels_full.detach().cpu().numpy()

        # Ensure labels are integers
        labels_train = labels_train.astype(int)
        labels_test = labels_test.astype(int)
        labels_full = labels_full.astype(int)

        # Apply t-SNE to the full, train, and test embeddings
        tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42)
        
        # t-SNE on full dataset
        embeddings_full_2d = tsne.fit_transform(embeddingFull)
        
        # t-SNE on train dataset
        embeddings_train_2d = tsne.fit_transform(embeddingTrain)
        
        # t-SNE on test dataset
        embeddings_test_2d = tsne.fit_transform(embeddingTest)

        # Map numeric labels to class names
        label_names_train = np.array([self.class_names[label] for label in labels_train])
        label_names_test = np.array([self.class_names[label] for label in labels_test])
        label_names_full = np.array([self.class_names[label] for label in labels_full])

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Full dataset plot
        palette = sns.color_palette("hls", len(np.unique(label_names_full)))
        sns.scatterplot(
            x=embeddings_full_2d[:, 0],
            y=embeddings_full_2d[:, 1],
            hue=label_names_full,
            palette=palette,
            legend='full',
            alpha=0.7,
            ax=axes[0]
        )
        axes[0].set_title(f'{title} - Full Dataset', fontsize=14)
        axes[0].set_xlabel('TSNE-1')
        axes[0].set_ylabel('TSNE-2')
        axes[0].grid(True)

        # Training set plot
        sns.scatterplot(
            x=embeddings_train_2d[:, 0],
            y=embeddings_train_2d[:, 1],
            hue=label_names_train,
            palette=palette,
            legend='full',
            alpha=0.7,
            ax=axes[1]
        )
        axes[1].set_title(f'{title} - Training Set', fontsize=14)
        axes[1].set_xlabel('TSNE-1')
        axes[1].set_ylabel('TSNE-2')
        axes[1].grid(True)

        # Test set plot
        sns.scatterplot(
            x=embeddings_test_2d[:, 0],
            y=embeddings_test_2d[:, 1],
            hue=label_names_test,
            palette=palette,
            legend='full',
            alpha=0.7,
            ax=axes[2]
        )
        axes[2].set_title(f'{title} - Test Set', fontsize=14)
        axes[2].set_xlabel('TSNE-1')
        axes[2].set_ylabel('TSNE-2')
        axes[2].grid(True)

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()




import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.manifold import TSNE

class Visualizer:
    def __init__(self, class_names):
        self.class_names = class_names
        # Create a fixed color mapping once
        self.color_mapping = self._generate_color_mapping()

    def _generate_color_mapping(self):
        palette = sns.color_palette("hls", len(self.class_names))
        return {class_name: palette[i] for i, class_name in enumerate(self.class_names)}

    def _convert_to_numpy(self, *arrays):
        converted = []
        for array in arrays:
            if isinstance(array, torch.Tensor):
                array = array.detach().cpu().numpy()
            converted.append(array)
        return converted

    def _map_labels(self, labels):
        return np.array([self.class_names[label] for label in labels])

    def _plot_embeddings(self, embeddings_list, labels_list, titles, axis_labels, title):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, (embeddings, label_names, subtitle) in enumerate(zip(embeddings_list, labels_list, titles)):
            colors = [self.color_mapping[label] for label in label_names]
            axes[idx].scatter(
                embeddings[:, 0], embeddings[:, 1],
                c=colors, alpha=0.7
            )
            axes[idx].set_title(f'{title} - {subtitle}', fontsize=14)
            axes[idx].set_xlabel(axis_labels[0])
            axes[idx].set_ylabel(axis_labels[1])
            axes[idx].grid(True)
        
        plt.tight_layout()
        plt.show()

    def visualize_embeddings_umap(self, embeddingTrain, embeddingTest, embeddingFull, 
                                   labels_train, labels_test, labels_full, title='UMAP Visualization'):
        
        embeddingTrain, embeddingTest, embeddingFull = self._convert_to_numpy(embeddingTrain, embeddingTest, embeddingFull)
        labels_train, labels_test, labels_full = self._convert_to_numpy(labels_train, labels_test, labels_full)

        labels_train = labels_train.astype(int)
        labels_test = labels_test.astype(int)
        labels_full = labels_full.astype(int)

        umap_model = umap.UMAP(n_components=2, random_state=42)

        embeddings_full_2d = umap_model.fit_transform(embeddingFull)
        embeddings_train_2d = umap_model.fit_transform(embeddingTrain)
        embeddings_test_2d = umap_model.fit_transform(embeddingTest)

        label_names_train = self._map_labels(labels_train)
        label_names_test = self._map_labels(labels_test)
        label_names_full = self._map_labels(labels_full)

        self._plot_embeddings(
            embeddings_list=[embeddings_full_2d, embeddings_train_2d, embeddings_test_2d],
            labels_list=[label_names_full, label_names_train, label_names_test],
            titles=["Full Dataset", "Training Set", "Test Set"],
            axis_labels=["UMAP-1", "UMAP-2"],
            title=title
        )

    def visualize_embeddings2(self, embeddingTrain, embeddingTest, embeddingFull, 
                               labels_train, labels_test, labels_full, title='t-SNE Visualization'):
        
        embeddingTrain, embeddingTest, embeddingFull = self._convert_to_numpy(embeddingTrain, embeddingTest, embeddingFull)
        labels_train, labels_test, labels_full = self._convert_to_numpy(labels_train, labels_test, labels_full)

        labels_train = labels_train.astype(int)
        labels_test = labels_test.astype(int)
        labels_full = labels_full.astype(int)

        tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42)

        embeddings_full_2d = tsne.fit_transform(embeddingFull)
        embeddings_train_2d = tsne.fit_transform(embeddingTrain)
        embeddings_test_2d = tsne.fit_transform(embeddingTest)

        label_names_train = self._map_labels(labels_train)
        label_names_test = self._map_labels(labels_test)
        label_names_full = self._map_labels(labels_full)

        self._plot_embeddings(
            embeddings_list=[embeddings_full_2d, embeddings_train_2d, embeddings_test_2d],
            labels_list=[label_names_full, label_names_train, label_names_test],
            titles=["Full Dataset", "Training Set", "Test Set"],
            axis_labels=["t-SNE-1", "t-SNE-2"],
            title=title
        )
