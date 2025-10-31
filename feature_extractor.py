# feature_extractor.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import timm
def get_resnet_features(X_numpy, device='cpu'):
    model = models.densenet201(pretrained=True)
    model.fc = torch.nn.Identity()  # remove final classification layer
    model.eval().to(device)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        features = []
        for img in X_numpy:
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)  # make 3 channels
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            feat = model(img_tensor)
            features.append(feat.cpu().numpy())
    
    return np.vstack(features)

def get_COnvNeXTfeatures(X_numpy, device='cpu'):
    model = timm.create_model('convnext_base', pretrained=True)
    model.fc = torch.nn.Identity()  # remove final classification layer
    model.eval().to(device)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        features = []
        for img in X_numpy:
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)  # make 3 channels
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            feat = model(img_tensor)
            features.append(feat.cpu().numpy())
    
    return np.vstack(features)

