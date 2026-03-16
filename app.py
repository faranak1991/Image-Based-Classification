#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, classification_report, 
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import io
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cpu')


test_image_path = "2.png" 


#======================================================================= MODEL CLASSES

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5, img_hw=224, 
                 conv_channels=[16, 32, 64], fc_hidden=256, dropout=0.2):
        super().__init__()
        
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        in_channels = 1
        for out_channels in conv_channels:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.bn_layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_hw, img_hw)
            for conv, bn in zip(self.conv_layers, self.bn_layers):
                dummy = self.pool(F.relu(bn(conv(dummy))))
            flat_img_feats = dummy.view(1, -1).size(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(flat_img_feats, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )
    
    def forward(self, x_img):
        x = x_img
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.pool(F.relu(bn(conv(x))))
        
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits


class DenseNet121(nn.Module):
    def __init__(self, num_classes=5, dropout=0.4, fc_hidden=256):
        super().__init__()
        backbone = models.densenet121(pretrained=False)
        
        old_conv = backbone.features.conv0
        new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride, padding=old_conv.padding, bias=False)
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        backbone.features.conv0 = new_conv
        
        self.features = backbone.features
        self.num_ftrs = backbone.classifier.in_features
        
        self.classifier = nn.Sequential(
            nn.Linear(self.num_ftrs, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )
    
    def forward(self, x_img):
        x = self.features(x_img)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


class ResNet18(nn.Module):
    def __init__(self, num_classes=5, dropout=0.4, fc_hidden=256, pretrained=True):
        super().__init__()
        backbone = models.resnet18(pretrained=False)
        
        old = backbone.conv1
        new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = new
        
        self.backbone = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu,
            backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.avgpool = backbone.avgpool
        self.backbone_out = backbone.fc.in_features
        
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone_out, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )
    
    def forward(self, x_img):
        x = self.backbone(x_img)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=5, dropout=0.4, fc_hidden=256, use_weights=True):
        super().__init__()
        
        backbone = models.efficientnet_b0(pretrained=False)
        
        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        backbone.features[0][0] = new_conv
        
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.num_ftrs = backbone.classifier[1].in_features
        
        self.classifier = nn.Sequential(
            nn.Linear(self.num_ftrs, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )
    
    def forward(self, x_img):
        x = self.features(x_img)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


#============================================================================ CUSTOM UNPICKLER

class CPU_Unpickler(pickle.Unpickler):
    """Custom unpickler that forces all tensors to CPU"""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


#============================================================================ IMAGE PREPROCESSING

def load_and_preprocess_image(img_path, img_size=(224, 224)):
    """Load and preprocess image EXACTLY as done in training"""
    try:
        pil_img = Image.open(img_path)
        cropped = pil_img.convert('L')
        cropped = cropped.resize(img_size)
        cropped = np.array(cropped, dtype=np.uint8)
        
        norm_clahe = cropped.astype(np.float32)
        norm_post = norm_clahe.astype(np.float32) / 255.0
        
        return norm_post
    
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return np.zeros(img_size, dtype=np.float32)


#============================================================================ PREDICTION FUNCTIONS

def prepare_model(model):
    """Unwrap DataParallel and move to CPU"""
    if hasattr(model, 'module'):
        model = model.module
    model = model.to(device)
    model.eval()
    return model


def get_nn_probabilities(model, X_data, batch_size=8):
    """Get probability predictions from neural network"""
    model.eval()
    X_tensor = torch.from_numpy(X_data).unsqueeze(1).float()
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_probs = []
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
    
    return np.vstack(all_probs)


def predict_single_image(img_path, available_models, ensemble_members, 
                        OPTIMIZED_WEIGHTS, CLASS_NAMES, IMG_SIZE):
    
    # Load and preprocess the image
    img = load_and_preprocess_image(img_path, IMG_SIZE)
    X_single = np.array([img], dtype=np.float32)
    
    # Get probabilities from each model
    model_probs = {}
    for model_name, model in available_models.items():
        probs = get_nn_probabilities(model, X_single)
        model_probs[model_name] = probs
    
    # Create weighted ensemble
    member_probs = []
    available_members = []
    
    for k in ensemble_members:
        if k in model_probs:
            member_probs.append(model_probs[k])
            available_members.append(k)
    
    if not member_probs:
        raise RuntimeError("No ensemble members found.")
    
    # Use only weights for available members
    available_indices = [ensemble_members.index(m) for m in available_members]
    weights = OPTIMIZED_WEIGHTS[available_indices]
    weights = weights / weights.sum()
    
    # Compute weighted average
    stacked = np.stack(member_probs, axis=0)
    avg_probs = np.tensordot(weights, stacked, axes=(0, 0))
    
    # Renormalize
    row_sums = avg_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    avg_probs = avg_probs / row_sums
    
    # Get prediction
    y_pred = np.argmax(avg_probs, axis=1)[0]
    predicted_label = CLASS_NAMES[y_pred]
    
    # Create probability dictionary
    class_probabilities = {CLASS_NAMES[i]: float(avg_probs[0, i]) 
                          for i in range(len(CLASS_NAMES))}
    
    return predicted_label, class_probabilities


#============================================================================ MAIN CODE

# Load checkpoint
with open("complete_checkpoint_optuna_optimized.pkl", "rb") as f:
    checkpoint = CPU_Unpickler(f).load()

# Extract configuration
config = checkpoint.get('config', {})
CLASS_NAMES = config.get('CLASS_NAMES', ['ADLs', 'Aggregates', 'Droplets', 'Gels', 'Solutions'])

DISPLAY_NAME_MAP = {"Solutions": "Solution"}
CLASS_NAMES = [DISPLAY_NAME_MAP.get(n, n) for n in CLASS_NAMES]

NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = config.get('IMG_SIZE', (224, 224))

le = checkpoint['label_encoder']
test_paths = checkpoint.get('test_paths', [])
final_results = checkpoint.get('final_results', {})

# Optimized weights
OPTIMIZED_WEIGHTS = np.array([0.279, 0.15685, 0.564], dtype=float)
ensemble_members = ['ResNet', 'EfficientNet', 'DenseNet']

#print("Loading ensemble models...")
available_models = {}
for model_name in ensemble_members:
    if model_name in final_results and 'model' in final_results[model_name]:
        model = final_results[model_name]['model']
        if model is not None:
            prepared = prepare_model(model)
            available_models[model_name] = prepared
           # print(f"✅ {model_name} loaded")

#print(f"\n✅ Ensemble ready with {len(available_models)} models")

#============================================================================ TEST ON SINGLE IMAGE

  

if os.path.exists(test_image_path):
   
    
    predicted_label, class_probs = predict_single_image(
        test_image_path, 
        available_models, 
        ensemble_members, 
        OPTIMIZED_WEIGHTS, 
        CLASS_NAMES, 
        IMG_SIZE
    )
    
    print(f"Predicted Label: {predicted_label}\n")
    #print("Class Probabilities:")
    print("-" * 40)
    
    # Sort by probability (descending)
    sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
    
    for class_name, prob in sorted_probs:
        bar = "█" * int(prob * 30)
        print(f"  {class_name:15s}  {prob*100:.2f}% {bar}")
    


