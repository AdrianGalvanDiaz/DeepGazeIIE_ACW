"""
This code was adapted from: https://github.com/rgeirhos/texture-vs-shape
"""
import os
import sys
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision
import torchvision.models
from torch.utils import model_zoo
import urllib.request
from pathlib import Path
import time

from .normalizer import Normalizer

# Alternate URLs or local paths for the models
MODELS_CONFIG = {
    'resnet50_trained_on_SIN': {
        'url': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
        'gdrive': 'https://drive.google.com/uc?id=1qX3HoyjU5NuKMAZaM3-DcwcP4cuXpqz9',
        'local_path': 'models/resnet50_train_60_epochs-c8e5653e.pth.tar'
    },
    'resnet50_trained_on_SIN_and_IN': {
        'url': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
        'gdrive': 'https://drive.google.com/uc?id=1JOxz47ZDEQz_ZIy3k_P9qfr_35RcZI3_',
        'local_path': 'models/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar'
    },
    'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': {
        'url': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
        'gdrive': 'https://drive.google.com/uc?id=1QMkp5_-kIQUAFVbMJf7sO3YCEyD_V8TV',
        'local_path': 'models/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar'
    }
}

def ensure_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

def download_file(url, filename, max_retries=3):
    for i in range(max_retries):
        try:
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            return True
        except Exception as e:
            print(f"Attempt {i+1} failed: {str(e)}")
            if i < max_retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Failed to download after {max_retries} attempts.")
                return False

def load_model(model_name):
    if model_name not in MODELS_CONFIG:
        raise ValueError(f"Unknown model: {model_name}")

    # Create models directory if it doesn't exist
    ensure_dir('models')
    
    config = MODELS_CONFIG[model_name]
    local_path = config['local_path']

    # If model doesn't exist locally, try to download it
    if not os.path.exists(local_path):
        # Try primary URL first
        success = download_file(config['url'], local_path)
        
        # If primary URL fails, try Google Drive backup
        if not success:
            print(f"Primary download failed. Please download the model manually from: {config['gdrive']}")
            print(f"Save it to: {os.path.abspath(local_path)}")
            raise ValueError(f"Could not download model {model_name}. Please download manually.")

    # Load the model architecture
    if "resnet50" in model_name:
        model = torchvision.models.resnet50(pretrained=False)
        model = torch.nn.Sequential(OrderedDict([('module', model)]))
    else:
        raise ValueError("Only ResNet50 models are currently supported")

    # Load the checkpoint
    try:
        checkpoint = torch.load(local_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["state_dict"])
        return model
    except Exception as e:
        print(f"Error loading model checkpoint: {str(e)}")
        raise

class RGBShapeNetA(nn.Sequential):
    def __init__(self):
        super(RGBShapeNetA, self).__init__()
        self.shapenet = load_model("resnet50_trained_on_SIN")
        self.normalizer = Normalizer()
        super(RGBShapeNetA, self).__init__(self.normalizer, self.shapenet)

class RGBShapeNetB(nn.Sequential):
    def __init__(self):
        super(RGBShapeNetB, self).__init__()
        self.shapenet = load_model("resnet50_trained_on_SIN_and_IN")
        self.normalizer = Normalizer()
        super(RGBShapeNetB, self).__init__(self.normalizer, self.shapenet)

class RGBShapeNetC(nn.Sequential):
    def __init__(self):
        super(RGBShapeNetC, self).__init__()
        self.shapenet = load_model("resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN")
        self.normalizer = Normalizer()
        super(RGBShapeNetC, self).__init__(self.normalizer, self.shapenet)