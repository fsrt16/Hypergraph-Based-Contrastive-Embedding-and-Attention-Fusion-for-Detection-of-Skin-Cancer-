import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import cv2
from tqdm import tqdm
import pandas as pd
def load_chest_xray_data(data_path, img_size=(256, 256)):
    X, Y = [], []
    categories = ['COVID19', 'NORMAL', 'PNEUMONIA']
    
    for category in categories:
        for phase in ['train', 'test']:
            path = os.path.join(data_path, phase, category)
            if not os.path.exists(path):
                continue  # skip if folder doesn't exist
            for img_file in tqdm(os.listdir(path), desc=f"{phase}/{category}"):
                img_path = os.path.join(path, img_file)
                try:
                    img = Image.open(img_path).convert('RGB')  # 🟢 Ensures 3 channels
                    img = img.resize(img_size)
                    img = np.array(img) / 255.0  # Normalize
                    X.append(img)
                    Y.append(category)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y)

    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)

    return X, Y, label_encoder.classes_



def load_skin_cancer_data(metadata_path, image_dir_p1, image_dir_p2, img_size=(224, 224)):
    """
    Load and preprocess skin cancer images and metadata.

    Args:
        metadata_path (str): Path to the CSV file containing metadata.
        image_dir_p1 (str): Path to part 1 of the image dataset.
        image_dir_p2 (str): Path to part 2 of the image dataset.
        img_size (tuple): Target image size (width, height).

    Returns:
        X (np.array): Preprocessed image data normalized [0,1].
        Y (np.array): Corresponding labels (diagnosis).
        df (pd.DataFrame): Full DataFrame with added 'path' column.
    """

    # Load metadata
    df = pd.read_csv(metadata_path)
    p1 = os.listdir(image_dir_p1)
    p2 = os.listdir(image_dir_p2)

    # Add image paths to dataframe
    df['path'] = ''
    for ind in df.index:
        img_filename = df.loc[ind, 'image_id'] + '.jpg'
        if img_filename in p1:
            df.loc[ind, 'path'] = os.path.join(image_dir_p1, img_filename)
        elif img_filename in p2:
            df.loc[ind, 'path'] = os.path.join(image_dir_p2, img_filename)

    # Load and preprocess images
    X, Y = [], []

    for idx in tqdm(df.index, desc="Loading images"):
        img_path = df.loc[idx, 'path']
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        X.append(img)
        Y.append(df.loc[idx, 'dx'])

    X = np.array(X, dtype=np.float32) / 255.0  # Normalize to [0,1]
    Y = np.array(Y)

    return X, Y, df
