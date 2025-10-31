from sklearn.preprocessing import LabelEncoder
import torch
from torchvision import models, transforms
# Model Trainer Class
# ----------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from timm import create_model
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay
from torchvision import models
import cv2
from timm import create_model
class DLClassifier2:
    def __init__(self, model_name, num_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = self.get_model(model_name, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def get_model(self, name, num_classes):
        


        model_registry = {
            'resnet18': lambda: models.resnet18(pretrained=True),
            'resnet34': lambda: models.resnet34(pretrained=True),
            'resnet50': lambda: models.resnet50(pretrained=True),
            'resnet101': lambda: models.resnet101(pretrained=True),
            'resnet152': lambda: models.resnet152(pretrained=True),
            
            'densenet121': lambda: models.densenet121(pretrained=True),
            'densenet169': lambda: models.densenet169(pretrained=True),
            'densenet201': lambda: models.densenet201(pretrained=True),
            'densenet161': lambda: models.densenet161(pretrained=True),

            'xception': lambda: create_model('xception', pretrained=True),
            'inceptionv3': lambda: models.inception_v3(pretrained=True),
            'inceptionv4': lambda: create_model('inception_v4', pretrained=True),
            'inceptionresnetv2': lambda: create_model('inception_resnet_v2', pretrained=True),

            'rexnet200': lambda: create_model('rexnet_200', pretrained=True),
            'res2next50': lambda: create_model('res2next50', pretrained=True),
            'mobilenetv3_large_100': lambda: create_model('mobilenetv3_large_100', pretrained=True),
            'mobilenetv2': lambda: models.mobilenet_v2(pretrained=True),
            
            'efficientnetb0': lambda: create_model('efficientnet_b0', pretrained=True),
            'efficientnetb1': lambda: create_model('efficientnet_b1', pretrained=True),
            'efficientnetb2': lambda: create_model('efficientnet_b2', pretrained=True),
            'efficientnetb3': lambda: create_model('efficientnet_b3', pretrained=True),
            'efficientnetb4': lambda: create_model('efficientnet_b4', pretrained=True),
            'efficientnetb5': lambda: create_model('efficientnet_b5', pretrained=True),
            'efficientnetb6': lambda: create_model('efficientnet_b6', pretrained=True),
            'efficientnetb7': lambda: create_model('efficientnet_b7', pretrained=True),
            'efficientnetv2_m': lambda: create_model('efficientnetv2_m', pretrained=True),
            'efficientnetv2_l': lambda: create_model('efficientnetv2_l', pretrained=True),
            'efficientnetv2_xl': lambda: create_model('efficientnetv2_xl', pretrained=True),

            'inceptionnext_base': lambda: create_model('inceptionnext_base', pretrained=True),
            'convnextv2_base': lambda: create_model('convnextv2_base', pretrained=True),
            
            'mobilevit_small': lambda: create_model('mobilevit_small', pretrained=True),
            'mobilevitv2_100': lambda: create_model('mobilevitv2_100', pretrained=True),
            
            'vit_base_patch16_224': lambda: create_model('vit_base_patch16_224', pretrained=True),
            'vit_base_patch16_384': lambda: create_model('vit_base_patch16_384', pretrained=True),
            'vit_large_patch16_224': lambda: create_model('vit_large_patch16_224', pretrained=True),
            'vit_large_patch16_384': lambda: create_model('vit_large_patch16_384', pretrained=True),
            'vit_huge_patch16_224': lambda: create_model('vit_huge_patch16_224', pretrained=True),

            'swin_base_patch4_window7_224': lambda: create_model('swin_base_patch4_window7_224', pretrained=True),
            'swin_large_patch4_window7_224': lambda: create_model('swin_large_patch4_window7_224', pretrained=True),
            'swin_base_patch4_window12_384': lambda: create_model('swin_base_patch4_window12_384', pretrained=True),
            
            'deit_base_patch16_224': lambda: create_model('deit_base_patch16_224', pretrained=True),
            'deit_base_patch16_384': lambda: create_model('deit_base_patch16_384', pretrained=True),
            'deit_small_patch16_224': lambda: create_model('deit_small_patch16_224', pretrained=True),
            
            'mvitv2_base': lambda: create_model('mvitv2_base', pretrained=True),
            'mvitv2_large': lambda: create_model('mvitv2_large', pretrained=True),

            'beitv2_base_patch16_224': lambda: create_model('beitv2_base_patch16_224', pretrained=True),
            
            'pit_base': lambda: create_model('pit_base', pretrained=True),
            'pit_large': lambda: create_model('pit_large', pretrained=True),
            
            'vitamin_base': lambda: create_model('vitamin_base', pretrained=True),
            'vitamin_large': lambda: create_model('vitamin_large', pretrained=True),

            'coat_lite_small': lambda: create_model('coat_lite_small', pretrained=True),
            'coat_lite_medium': lambda: create_model('coat_lite_medium', pretrained=True),
            'coat_lite_large': lambda: create_model('coat_lite_large', pretrained=True),

            # Added additional models
            'vgg11': lambda: models.vgg11(pretrained=True),
            'vgg13': lambda: models.vgg13(pretrained=True),
            'vgg16': lambda: models.vgg16(pretrained=True),
            'vgg19': lambda: models.vgg19(pretrained=True),
            'alexnet': lambda: models.alexnet(pretrained=True),
            'googlenet': lambda: models.googlenet(pretrained=True),
            'shufflenet_v2_x0_5': lambda: create_model('shufflenet_v2_x0_5', pretrained=True),
            'shufflenet_v2_x1_0': lambda: create_model('shufflenet_v2_x1_0', pretrained=True),
            'shufflenet_v2_x1_5': lambda: create_model('shufflenet_v2_x1_5', pretrained=True),
            'shufflenet_v2_x2_0': lambda: create_model('shufflenet_v2_x2_0', pretrained=True),
            'resnest50': lambda: create_model('resnest50', pretrained=True),
            'resnest101': lambda: create_model('resnest101', pretrained=True),
            'resnest200': lambda: create_model('resnest200', pretrained=True),
            'resnest269': lambda: create_model('resnest269', pretrained=True),
        }


        model = model_registry[name]()

        # Try to identify and replace the classification head
        if hasattr(model, 'fc'):  # e.g., ResNet
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)

        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):  # e.g., MobileNet, EfficientNet
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)

        elif hasattr(model, 'head') and isinstance(model.head, nn.Linear):  # e.g., ViT
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)

        elif isinstance(model, nn.Sequential):
            # Try to replace the last layer of the Sequential model
            last_layer = list(model.children())[-1]
            if isinstance(last_layer, nn.Linear):
                in_features = last_layer.in_features
                model[-1] = nn.Linear(in_features, num_classes)
            else:
                raise ValueError(f"Unsupported last layer type in Sequential model: {type(last_layer)}")

        else:
            raise ValueError(f"Unable to identify classification head for model {name}")

        return model


    def train(self, X, Y, real_classes, epochs=10, batch_size=32):
        # Prepare dataset
        dataset = CustomImageDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    def validate(self, X, Y, real_classes, batch_size=32):
        dataset = CustomImageDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        all_preds, all_true, all_probs = [], [], []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_true.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        print("\nClassification Report:")
        print(classification_report(all_true, all_preds, target_names=real_classes))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(all_true, all_preds)
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=real_classes, yticklabels=real_classes, cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

        # ROC Curve (for multi-class)
        Y_true = label_binarize(all_true, classes=list(range(len(real_classes))))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(real_classes)):
            fpr[i], tpr[i], _ = roc_curve(Y_true[:, i], np.array(all_probs)[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10, 6))
        for i in range(len(real_classes)):
            plt.plot(fpr[i], tpr[i], label=f"Class {real_classes[i]} (AUC = {roc_auc[i]:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Multi-class')
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert tensor to PIL Image if it's not already
    transforms.Resize((256, 256)),  # Resize to the correct dimensions
    transforms.ToTensor(),  # Convert image to tensor of shape (C, H, W)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.X = torch.tensor(self.X, dtype=torch.float)
        self.X = self.X.permute(0, 3, 1, 2)
        self.Y = Y
        self.transform = transform

        # Encode string labels into integers
        self.label_encoder = LabelEncoder()
        self.Y_encoded = self.label_encoder.fit_transform(self.Y)  # Encode labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        label = self.Y_encoded[idx]  # Use the encoded label

        # Apply transformation (resize, normalize) if provided
        if self.transform:
            img = self.transform(img)

        # Convert the label to tensor of type long (for classification)
        return img, torch.tensor(label, dtype=torch.long)