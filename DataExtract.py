import os
import sys
import subprocess

def install_if_missing(package_name: str):
    """Install a Python package via pip if it is not already installed."""
    try:
        __import__(package_name)
    except ImportError:
        print(f"{package_name} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# 1. Ensure kagglehub is installed
install_if_missing("kagglehub")

# 2. Import kagglehub
import kagglehub

# 3. (Optional) set a custom cache directory where dataset will be stored
# By default, kagglehub uses ~/.cache/kagglehub
# Uncomment and edit the next line if you want a specific location:
# os.environ["KAGGLEHUB_CACHE"] = "/path/where/you/want/to/save"

# 4. Download latest version of the HAM10000 dataset
dataset_id = "kmader/skin-cancer-mnist-ham10000"

print(f"Downloading dataset: {dataset_id} ...")
path = kagglehub.dataset_download(dataset_id)

# 5. Print the path containing the downloaded files
print("Path to dataset files:", path)

# 6. (Optional) list the files that were downloaded
for root, dirs, files in os.walk(path):
    for fname in files:
        print(os.path.join(root, fname))
      
