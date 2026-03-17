import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import cv2
from PIL import Image
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm
import seaborn as sns
import albumentations as A
from sklearn.utils import shuffle
import random
from collections import defaultdict, Counter
import pandas as pd
import pickle
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Optuna imports
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import sys
class UnbufferedOutput:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def flush(self):
        self.stream.flush()
sys.stdout = UnbufferedOutput(open("output.txt", "w"))
sys.stderr = sys.stdout

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global Configuration
IMG_SIZE = (224, 224)
CLASS_NAMES = ["ADLs", "Aggregates", "Droplets", "Gels", "Solutions"]
NUM_CLASSES = len(CLASS_NAMES)
TEST_SIZE = 0.15
RANDOM_STATE = 42
DESIRED_CLASS_SIZE = 600
BATCH_SIZE = 8
NUM_EPOCHS = 200
PATIENCE = 20
LEARNING_RATE = 1e-4

# Optuna Configuration
N_TRIALS = 200  # Number of hyperparameter trials per model
OPTUNA_TIMEOUT = 7200  # 2 hours timeout per model

# Device setup
def setup_device():
    """Setup device for distributed training if available"""
    if 'WORLD_SIZE' in os.environ:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    return device

# Utility functions
def texture_focused_crop(img, target_size=(500, 500)):
    """Crop the region with highest texture entropy for better feature extraction"""
    gray = np.array(img.convert('L'))
    
    crop_w, crop_h = target_size
    img_h, img_w = gray.shape
    
    if img_h < crop_h or img_w < crop_w:
        return img
    
    best_entropy = 0
    best_crop = None
    
    step = 20
    for y in range(0, img_h - crop_h, step):
        for x in range(0, img_w - crop_w, step):
            window = gray[y:y+crop_h, x:x+crop_w]
            
            hist = np.histogram(window, bins=64, range=(0, 256))[0]
            hist = hist[hist > 0]
            if len(hist) > 1:
                prob = hist / hist.sum()
                entropy = -np.sum(prob * np.log2(prob))
                
                if entropy > best_entropy:
                    best_entropy = entropy
                    best_crop = (x, y, x + crop_w, y + crop_h)
    
    if best_crop:
        return img.crop(best_crop)
    else:
        width, height = img.size
        left = (width - crop_w) // 2
        top = (height - crop_h) // 2
        return img.crop((left, top, left + crop_w, top + crop_h))

def calculate_per_class_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    per_class_acc = {}
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    
    for i, class_name in enumerate(class_names):
        if cm[i].sum() > 0:  # Avoid division by zero
            per_class_acc[class_name] = cm[i, i] / cm[i].sum()
        else:
            per_class_acc[class_name] = 0.0
    
    return per_class_acc

def analyze_misclassifications_traditional(model, X_test, y_test, paths_test, le, model_name):
    """Analyze misclassifications for traditional ML models"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    misclassified_info = []
    for i in range(len(y_test)):
        if y_pred[i] != y_test[i]:
            record = {
                "Image Name": os.path.basename(paths_test[i]),
                "True Label": le.classes_[y_test[i]],
                "Predicted Label": le.classes_[y_pred[i]],
            }
            # Add probabilities for each class
            for j, class_name in enumerate(le.classes_):
                record[f"Prob_{class_name}"] = y_pred_proba[i][j]
            misclassified_info.append(record)
    
    df_misclassified = pd.DataFrame(misclassified_info)
    filename = f"misclassified_images_{model_name.lower()}.csv"
    df_misclassified.to_csv(filename, index=False)
    print(f"Saved misclassified image report to: {filename}")
    return df_misclassified

def analyze_misclassifications_deep(model, test_loader, paths_test, le, model_name, device):
    """Analyze misclassifications for deep learning models"""
    model.eval()
    misclassified_info = []
    sample_idx = 0
    
    with torch.no_grad():
        for X_img_batch, y_batch in test_loader:
            batch_size = X_img_batch.size(0)
            X_img_batch = X_img_batch.to(device)
            
            outputs = model(X_img_batch)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            y_batch_np = y_batch.numpy()
            
            for i in range(batch_size):
                if preds[i] != y_batch_np[i]:
                    record = {
                        "Image Name": os.path.basename(paths_test[sample_idx + i]),
                        "True Label": le.classes_[y_batch_np[i]],
                        "Predicted Label": le.classes_[preds[i]],
                    }
                    # Add probabilities for each class
                    for j, class_name in enumerate(le.classes_):
                        record[f"Prob_{class_name}"] = probs[i][j]
                    misclassified_info.append(record)
            sample_idx += batch_size
    
    df_misclassified = pd.DataFrame(misclassified_info)
    filename = f"misclassified_images_{model_name.lower()}.csv"
    df_misclassified.to_csv(filename, index=False)
    print(f"Saved misclassified image report to: {filename}")
    return df_misclassified

# Data loading and preprocessing
print("Loading and preprocessing images with cropping...")

X_img, y, image_paths_all = [], [], []
class_counts = {name: 0 for name in CLASS_NAMES}

for class_name in CLASS_NAMES:
    class_path = os.path.join(class_name)
    if not os.path.exists(class_path):
        print(f"Warning: Class directory {class_path} not found!")
        continue

    image_paths = glob(os.path.join(class_path, "*.png"))

    for img_path in image_paths:
        try:
            pil_img = Image.open(img_path)
            cropped = pil_img.convert('L')
            cropped = cropped.resize(IMG_SIZE)
            cropped = np.array(cropped, dtype=np.uint8)

          #  clahe = cv2.createCLAHE(clipLimit=6, tileGridSize=(3, 3))
           # norm_clahe = clahe.apply(cropped)
            norm_clahe = cropped.astype(np.float32)
            norm_post = norm_clahe.astype(np.float32) / 255.0

            X_img.append(norm_post)
            y.append(class_name)
            image_paths_all.append(img_path)
            class_counts[class_name] += 1

        except Exception as e:
            print(f"Skipping {img_path}: {e}")
            continue

# Convert to numpy arrays
X = np.stack(X_img)
y = np.array(y)

if len(X) == 0:
    raise RuntimeError("No images were loaded. Check CLASS_NAMES directories and file patterns.")

# Encode class labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\nLoaded {len(X)} images total")
for cls, count in class_counts.items():
    print(f"{cls}: {count} images")

# Train/test split
X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
    X, y_encoded, image_paths_all,
    test_size=TEST_SIZE, stratify=y_encoded, random_state=RANDOM_STATE
)

# Data augmentation
print("\nApplying data augmentation...")

augment_images = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.Resize(*IMG_SIZE)
])

class_to_images = defaultdict(list)

for img, label in zip(X_train, y_train):
    class_to_images[label].append(img)

X_train_balanced, y_train_balanced = [], []

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

for label_id in range(NUM_CLASSES):
    images = class_to_images[label_id]
    current_count = len(images)

    X_train_balanced.extend(images)
    y_train_balanced.extend([label_id] * current_count)

    if current_count < DESIRED_CLASS_SIZE:
        needed = DESIRED_CLASS_SIZE - current_count
        for _ in range(needed):
            idx = random.randint(0, len(images) - 1)
            img_aug = augment_images(image=images[idx])['image']
            X_train_balanced.append(img_aug)
            y_train_balanced.append(label_id)

X_train_balanced = np.asarray(X_train_balanced, dtype=np.float32)
y_train_balanced = np.asarray(y_train_balanced, dtype=np.int64)

indices = np.random.RandomState(RANDOM_STATE).permutation(len(X_train_balanced))
X_train = X_train_balanced[indices]
y_train = y_train_balanced[indices]

# Prepare data for traditional ML
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

device = setup_device()

# =============================================================================
# OPTUNA HYPERPARAMETER OPTIMIZATION
# =============================================================================

def objective_random_forest(trial):
    """Optuna objective function for Random Forest"""
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    
    # Create model with suggested hyperparameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # Use cross-validation for robust evaluation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
    
    return cv_scores.mean()


# CNN Model Classes
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5, img_hw=224, 
                 conv_channels=[16, 32, 64], fc_hidden=256, dropout=0.2):
        super().__init__()
        
        # Configurable conv layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        in_channels = 1
        for out_channels in conv_channels:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.bn_layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        
        # Calculate flattened size
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
        backbone = models.densenet121(pretrained=True)
        
        # Change first conv to 1 input channel
        old_conv = backbone.features.conv0
        new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride, padding=old_conv.padding, bias=False)
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        backbone.features.conv0 = new_conv
        
        self.features = backbone.features
        self.num_ftrs = backbone.classifier.in_features
        
        # Classifier
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
        backbone = models.resnet18(pretrained=pretrained)
        
        # Replace first conv
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
        
        # Classifier
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
        
        if use_weights and hasattr(models, "EfficientNet_B0_Weights"):
            backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            backbone = models.efficientnet_b0(pretrained=True)
        
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
        
        # Classifier
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

# Deep learning objective functions
def train_and_evaluate_model(model, train_loader, val_loader, lr, epochs=50):
    """Train a model and return validation F1 score"""
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_f1 = 0.0
    patience_counter = 0
    patience = 10  # Reduced for faster hyperparameter search
    
    for epoch in range(epochs):
        # Training
        model.train()
        for X_img, y in train_loader:
            X_img = X_img.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            logits = model(X_img)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_img, y in val_loader:
                X_img = X_img.to(device)
                logits = model(X_img)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(y.numpy())
        
        f1 = f1_score(y_true, y_pred, average='macro')
        
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_f1

def objective_cnn(trial):
    """Optuna objective function for Simple CNN"""
    # Suggest hyperparameters
    conv_channels = []
    n_conv_layers = trial.suggest_int('n_conv_layers', 2, 4)
    
    for i in range(n_conv_layers):
        channels = trial.suggest_categorical(f'conv_channels_{i}', [16, 32, 64, 128, 256])
        conv_channels.append(channels)
    
    fc_hidden = trial.suggest_categorical('fc_hidden', [128, 256, 512])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    
    # Create data loaders with suggested batch size
    X_train_tensor = torch.from_numpy(X_train).unsqueeze(1).float()
    X_test_tensor = torch.from_numpy(X_test).unsqueeze(1).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    y_test_tensor = torch.from_numpy(y_test).long()
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = SimpleCNN(
        num_classes=NUM_CLASSES,
        img_hw=IMG_SIZE[0],
        conv_channels=conv_channels,
        fc_hidden=fc_hidden,
        dropout=dropout
    ).to(device)
    
    return train_and_evaluate_model(model, train_loader, val_loader, lr, epochs=50)

def objective_densenet(trial):
    """Optuna objective function for DenseNet"""
    # Suggest hyperparameters
    fc_hidden = trial.suggest_categorical('fc_hidden', [128, 256, 512, 1024])
    dropout = trial.suggest_float('dropout', 0.2, 0.6)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    
    # Create data loaders
    X_train_tensor = torch.from_numpy(X_train).unsqueeze(1).float()
    X_test_tensor = torch.from_numpy(X_test).unsqueeze(1).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    y_test_tensor = torch.from_numpy(y_test).long()
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = DenseNet121(
        num_classes=NUM_CLASSES,
        dropout=dropout,
        fc_hidden=fc_hidden
    ).to(device)
    
    return train_and_evaluate_model(model, train_loader, val_loader, lr, epochs=50)

def objective_resnet(trial):
    """Optuna objective function for ResNet"""
    # Suggest hyperparameters
    fc_hidden = trial.suggest_categorical('fc_hidden', [128, 256, 512, 1024])
    dropout = trial.suggest_float('dropout', 0.2, 0.6)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    
    # Create data loaders
    X_train_tensor = torch.from_numpy(X_train).unsqueeze(1).float()
    X_test_tensor = torch.from_numpy(X_test).unsqueeze(1).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    y_test_tensor = torch.from_numpy(y_test).long()
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = ResNet18(
        num_classes=NUM_CLASSES,
        dropout=dropout,
        fc_hidden=fc_hidden,
        pretrained=True
    ).to(device)
    
    return train_and_evaluate_model(model, train_loader, val_loader, lr, epochs=50)

def objective_efficientnet(trial):
    """Optuna objective function for EfficientNet"""
    # Suggest hyperparameters
    fc_hidden = trial.suggest_categorical('fc_hidden', [128, 256, 512, 1024])
    dropout = trial.suggest_float('dropout', 0.2, 0.6)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    
    # Create data loaders
    X_train_tensor = torch.from_numpy(X_train).unsqueeze(1).float()
    X_test_tensor = torch.from_numpy(X_test).unsqueeze(1).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    y_test_tensor = torch.from_numpy(y_test).long()
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = EfficientNetB0(
        num_classes=NUM_CLASSES,
        dropout=dropout,
        fc_hidden=fc_hidden,
        use_weights=True
    ).to(device)
    
    return train_and_evaluate_model(model, train_loader, val_loader, lr, epochs=50)

# =============================================================================
# RUN HYPERPARAMETER OPTIMIZATION
# =============================================================================

print("\n" + "="*80)
print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
print("="*80)

# Create study configurations
study_configs = {
    'RandomForest': {
        'objective': objective_random_forest,
        'direction': 'maximize',
        'sampler': TPESampler(seed=RANDOM_STATE),
        'pruner': None
    },
    'SimpleCNN': {
        'objective': objective_cnn,
        'direction': 'maximize',
        'sampler': TPESampler(seed=RANDOM_STATE),
        'pruner': MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    },
    'DenseNet': {
        'objective': objective_densenet,
        'direction': 'maximize', 
        'sampler': TPESampler(seed=RANDOM_STATE),
        'pruner': MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    },
    'ResNet': {
        'objective': objective_resnet,
        'direction': 'maximize',
        'sampler': TPESampler(seed=RANDOM_STATE),
        'pruner': MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    },
    'EfficientNet': {
        'objective': objective_efficientnet,
        'direction': 'maximize',
        'sampler': TPESampler(seed=RANDOM_STATE), 
        'pruner': MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    }
}

# Store best hyperparameters and scores
best_hyperparams = {}
optimization_results = {}

for model_name, config in study_configs.items():
    print(f"\n{'='*60}")
    print(f"OPTIMIZING {model_name.upper()}")
    print(f"{'='*60}")
    
    # Create study
    study = optuna.create_study(
        direction=config['direction'],
        sampler=config['sampler'],
        pruner=config['pruner'],
        study_name=f"{model_name}_optimization"
    )
    
    # Run optimization
    try:
        study.optimize(
            config['objective'],
            n_trials=N_TRIALS,
            timeout=OPTUNA_TIMEOUT,
            show_progress_bar=True
        )
        
        # Store results
        best_hyperparams[model_name] = study.best_params
        optimization_results[model_name] = {
            'best_score': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'study': study
        }
        
        print(f"\n🏆 Best {model_name} F1 Score: {study.best_value:.4f}")
        print(f"🔧 Best parameters:")
        for param, value in study.best_params.items():
            print(f"   {param}: {value}")
            
    except Exception as e:
        print(f"❌ Error optimizing {model_name}: {e}")
        continue

# =============================================================================
# TRAIN FINAL MODELS WITH OPTIMIZED HYPERPARAMETERS  
# =============================================================================

print("\n" + "="*80)
print("TRAINING FINAL MODELS WITH OPTIMIZED HYPERPARAMETERS")
print("="*80)

final_results = {}

# 1. Random Forest with optimized hyperparameters
if 'RandomForest' in best_hyperparams:
    print(f"\nTraining RandomForest with optimized hyperparameters...")  # CHANGED: Removed space
    rf_params = best_hyperparams['RandomForest']
    
    rf_model = RandomForestClassifier(
        **rf_params,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    y_pred_rf = rf_model.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf, average='macro')
    
    # CALCULATE AND PRINT PER-CLASS ACCURACY FOR RANDOM FOREST
    rf_per_class_acc = calculate_per_class_accuracy(y_test, y_pred_rf, CLASS_NAMES)
    print(f"RandomForest - Overall Accuracy: {rf_acc:.4f}, F1: {rf_f1:.4f}")  # CHANGED: Removed space
    print("RandomForest - Per-class accuracy:")  # CHANGED: Removed space
    for class_name, acc in rf_per_class_acc.items():
        print(f"  {class_name}: {acc:.4f}")
    
    # ADD THIS: Print classification report for precision, recall, F1-score
    print("\nClassification Report: RandomForest")
    print(classification_report(y_test, y_pred_rf, target_names=CLASS_NAMES))
    
    # ANALYZE MISCLASSIFICATIONS FOR RANDOM FOREST
    rf_misclassified_df = analyze_misclassifications_traditional(rf_model, X_test_scaled, y_test, paths_test, le, "RandomForest")

    final_results['RandomForest'] = {
    'accuracy': rf_acc,
    'f1_score': rf_f1,
    'hyperparameters': rf_params,
    'predictions': y_pred_rf,
    'model': rf_model,  
    'per_class_accuracy': rf_per_class_acc,
    'misclassified_df': rf_misclassified_df
    }
    
    
# Function to train deep learning model with best hyperparameters
def train_final_deep_model(model_class, model_name, best_params):
    """Train a deep learning model with optimized hyperparameters for full epochs"""
    print(f"\nTraining {model_name} with optimized hyperparameters...")
    
    # Extract hyperparameters
    batch_size = best_params.get('batch_size', 8)
    lr = best_params.get('lr', 1e-4)
    dropout = best_params.get('dropout', 0.4)
    fc_hidden = best_params.get('fc_hidden', 256)
    
    # Create data loaders
    X_train_tensor = torch.from_numpy(X_train).unsqueeze(1).float()
    X_test_tensor = torch.from_numpy(X_test).unsqueeze(1).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    y_test_tensor = torch.from_numpy(y_test).long()
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model with optimized hyperparameters
    if model_name == 'SimpleCNN':
        conv_channels = []
        n_conv_layers = best_params.get('n_conv_layers', 3)
        for i in range(n_conv_layers):
            conv_channels.append(best_params.get(f'conv_channels_{i}', [16, 32, 64][i % 3]))
        
        model = model_class(
            num_classes=NUM_CLASSES,
            img_hw=IMG_SIZE[0],
            conv_channels=conv_channels,
            fc_hidden=fc_hidden,
            dropout=dropout
        ).to(device)
    else:
        # For pretrained models (DenseNet, ResNet, EfficientNet)
        model_kwargs = {
            'num_classes': NUM_CLASSES,
            'dropout': dropout,
            'fc_hidden': fc_hidden
        }
        
        if model_name in ['ResNet', 'EfficientNet']:
            if model_name == 'ResNet':
                model_kwargs['pretrained'] = True
            elif model_name == 'EfficientNet':
                model_kwargs['use_weights'] = True
        
        model = model_class(**model_kwargs).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop with full epochs
    train_losses, test_accuracies, f1_scores = [], [], []
    best_f1, patience_counter = 0.0, 0
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for X_img, y in train_loader:
            X_img = X_img.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            logits = model(X_img)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluation phase
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_img, y in test_loader:
                X_img = X_img.to(device)
                logits = model(X_img)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(y.numpy())
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        test_accuracies.append(acc)
        f1_scores.append(f1)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f} - Acc: {acc:.4f} - F1: {f1:.4f}")
        
        # Early stopping
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model_{model_name.lower()}_optimized.pth")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}. Best F1: {best_f1:.4f}")
                break
    
    # Load best model and final evaluation (ONLY ONCE!)
    model.load_state_dict(torch.load(f"best_model_{model_name.lower()}_optimized.pth"))
    model.eval()
    
    y_true_final, y_pred_final = [], []
    with torch.no_grad():
        for X_img, y in test_loader:
            X_img = X_img.to(device)
            logits = model(X_img)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred_final.extend(preds)
            y_true_final.extend(y.numpy())
    
    final_acc = accuracy_score(y_true_final, y_pred_final)
    final_f1 = f1_score(y_true_final, y_pred_final, average='macro')
    
    # CALCULATE AND PRINT PER-CLASS ACCURACY FOR DEEP LEARNING MODELS
    per_class_acc = calculate_per_class_accuracy(y_true_final, y_pred_final, CLASS_NAMES)
    print(f"{model_name} - Overall Accuracy: {final_acc:.4f}, F1: {final_f1:.4f}")
    print(f"{model_name} - Per-class accuracy:")
    for class_name, acc in per_class_acc.items():
        print(f"  {class_name}: {acc:.4f}")
    
    # ADD THIS: Print classification report for precision, recall, F1-score
    print(f"\nClassification Report: {model_name}")
    print(classification_report(y_true_final, y_pred_final, target_names=CLASS_NAMES))
    
    # ANALYZE MISCLASSIFICATIONS FOR DEEP LEARNING MODELS
    misclassified_df = analyze_misclassifications_deep(model, test_loader, paths_test, le, model_name, device)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true_final, y_pred_final, labels=list(range(NUM_CLASSES)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title(f"Confusion Matrix ({model_name} - Optimized)")
    plt.tight_layout()
    plt.savefig(f"cm_{model_name}_optimized.png", dpi=300)
    plt.close()
    
    # Training curves
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-')
    plt.title(f"{model_name} Training Loss (Optimized)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, test_accuracies, 'g-')
    plt.title(f"{model_name} Test Accuracy (Optimized)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, f1_scores, 'r-')
    plt.title(f"{model_name} Test F1 Score (Optimized)")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{model_name.lower()}_optimized_training_metrics.png", dpi=300)
    plt.close()
    
    return {
        'accuracy': final_acc,
        'f1_score': final_f1,
        'hyperparameters': best_params,
        'predictions': y_pred_final,
        'model': model,
        'per_class_accuracy': per_class_acc,
        'misclassified_df': misclassified_df,
        'training_history': {
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'f1_scores': f1_scores
        }
    }

# Train deep learning models with optimized hyperparameters
deep_model_classes = {
    'SimpleCNN': SimpleCNN,
    'DenseNet': DenseNet121,
    'ResNet': ResNet18,
    'EfficientNet': EfficientNetB0
}

for model_name, model_class in deep_model_classes.items():
    if model_name in best_hyperparams:
        result = train_final_deep_model(model_class, model_name, best_hyperparams[model_name])
        final_results[model_name] = result

# =============================================================================
# RESULTS SUMMARY AND COMPARISON
# =============================================================================

print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

# Create results comparison
results_df = []
for model_name, results in final_results.items():
    results_df.append({
        'Model': model_name,
        'Test Accuracy': results['accuracy'],
        'Test F1 Score': results['f1_score'],
        'Hyperparameters': str(results['hyperparameters'])
    })

results_comparison = pd.DataFrame(results_df)
results_comparison = results_comparison.sort_values('Test F1 Score', ascending=False)

print("\nModel Performance Comparison (sorted by F1 score):")
print("="*60)
for _, row in results_comparison.iterrows():
    print(f"{row['Model']:<20} | Acc: {row['Test Accuracy']:.4f} | F1: {row['Test F1 Score']:.4f}")

# PRINT PER-CLASS ACCURACY SUMMARY FOR ALL MODELS
print("\n" + "="*80)
print("PER-CLASS ACCURACY SUMMARY FOR ALL MODELS")
print("="*80)
for model_name, results in final_results.items():
    print(f"\n{model_name}:")
    print("-" * (len(model_name) + 1))
    if 'per_class_accuracy' in results:
        for class_name, acc in results['per_class_accuracy'].items():
            print(f"  {class_name}: {acc:.4f}")
    print(f"  Overall: {results['accuracy']:.4f}")

# PRINT MISCLASSIFICATION SUMMARY
print("\n" + "="*80)
print("MISCLASSIFICATION ANALYSIS SUMMARY")
print("="*80)
for model_name, results in final_results.items():
    if 'misclassified_df' in results:
        num_misclassified = len(results['misclassified_df'])
        total_test_samples = len(y_test)
        print(f"\n{model_name}:")
        print("-" * (len(model_name) + 1))
        print(f"  Total misclassified: {num_misclassified}/{total_test_samples} ({num_misclassified/total_test_samples*100:.1f}%)")
        
        if num_misclassified > 0:
            # Show most common misclassification patterns
            misclass_patterns = results['misclassified_df'].groupby(['True Label', 'Predicted Label']).size().sort_values(ascending=False)
            print(f"  Most common misclassification patterns:")
            for i, ((true_label, pred_label), count) in enumerate(misclass_patterns.head(3).items()):
                print(f"    {true_label} -> {pred_label}: {count} images")

# Save detailed results
results_comparison.to_csv("model_comparison_optimized.csv", index=False)

# Plot model comparison
plt.figure(figsize=(12, 6))

models = results_comparison['Model'].values
accuracies = results_comparison['Test Accuracy'].values
f1_scores_arr = results_comparison['Test F1 Score'].values

x = np.arange(len(models))
width = 0.35

plt.subplot(1, 2, 1)
bars1 = plt.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
bars2 = plt.bar(x + width/2, f1_scores_arr, width, label='F1 Score', alpha=0.8)

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison (Optimized)')
plt.xticks(x, models, rotation=45)
plt.legend()
plt.ylim(0, 1.05)
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=8)

for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.subplot(1, 2, 2)
sorted_models = results_comparison.sort_values('Test F1 Score', ascending=True)
plt.barh(sorted_models['Model'], sorted_models['Test F1 Score'], color='skyblue', alpha=0.8)
plt.xlabel('F1 Score')
plt.title('Models Ranked by F1 Score')
plt.xlim(0, 1.05)
plt.grid(axis='x', alpha=0.3)

for i, (model, score) in enumerate(zip(sorted_models['Model'], sorted_models['Test F1 Score'])):
    plt.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig("model_comparison_optimized.png", dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# OPTUNA VISUALIZATION AND ANALYSIS
# =============================================================================

print("\nGenerating Optuna optimization visualizations...")

# Create optimization history plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (model_name, results) in enumerate(optimization_results.items()):
    if i >= 6:  # Max 6 subplots
        break
        
    study = results['study']
    trials = study.trials
    
    # Plot optimization history
    trial_numbers = [t.number for t in trials]
    values = [t.value for t in trials if t.value is not None]
    
    if len(values) > 0:
        axes[i].plot(trial_numbers[:len(values)], values, 'b-', alpha=0.7)
        axes[i].axhline(y=study.best_value, color='r', linestyle='--', 
                       label=f'Best: {study.best_value:.4f}')
        axes[i].set_title(f'{model_name} Optimization History')
        axes[i].set_xlabel('Trial')
        axes[i].set_ylabel('F1 Score')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

# Hide empty subplots
for j in range(len(optimization_results), 6):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig("optuna_optimization_history.png", dpi=300, bbox_inches='tight')
plt.close()

# Create hyperparameter importance plots
importance_data = {}
for model_name, results in optimization_results.items():
    study = results['study']
    try:
        importance = optuna.importance.get_param_importances(study)
        importance_data[model_name] = importance
    except Exception as e:
        print(f"Could not compute parameter importance for {model_name}: {e}")

if importance_data:
    n_models = len(importance_data)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 6))
    if n_models == 1:
        axes = [axes]
    
    for i, (model_name, importance) in enumerate(importance_data.items()):
        params = list(importance.keys())
        values = list(importance.values())
        
        axes[i].barh(params, values)
        axes[i].set_title(f'{model_name}\nHyperparameter Importance')
        axes[i].set_xlabel('Importance')
        axes[i].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("hyperparameter_importance.png", dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# SAVE COMPREHENSIVE RESULTS
# =============================================================================

# Save complete checkpoint with Optuna results
checkpoint = {
    'config': {
        'IMG_SIZE': IMG_SIZE,
        'CLASS_NAMES': CLASS_NAMES,
        'NUM_CLASSES': NUM_CLASSES,
        'cropping_enabled': True,
        'optuna_trials': N_TRIALS,
        'optuna_timeout': OPTUNA_TIMEOUT
    },
    'label_encoder': le,
    'scaler': scaler,
    'optuna_results': {
        'best_hyperparameters': best_hyperparams,
        'optimization_results': {k: {
            'best_score': v['best_score'],
            'best_params': v['best_params'],
            'n_trials': v['n_trials']
        } for k, v in optimization_results.items()},
    },
    'final_results': final_results,
    'test_paths': paths_test,
    'class_counts': class_counts
}

with open("complete_checkpoint_optuna_optimized.pkl", "wb") as f:
    pickle.dump(checkpoint, f)

# Save hyperparameters to separate file for easy reference
with open("best_hyperparameters.json", "w") as f:
    import json
    json.dump(best_hyperparams, f, indent=2)

# Create detailed report
report_lines = [
    "HYPERPARAMETER OPTIMIZATION REPORT",
    "="*50,
    f"Optimization completed with {N_TRIALS} trials per model",
    f"Timeout: {OPTUNA_TIMEOUT/3600:.1f} hours per model",
    "",
    "BEST HYPERPARAMETERS:",
    "-"*30
]

for model_name, params in best_hyperparams.items():
    report_lines.append(f"\n{model_name}:")
    for param, value in params.items():
        report_lines.append(f"  {param}: {value}")

report_lines.extend([
    "",
    "FINAL PERFORMANCE:",
    "-"*30
])

for _, row in results_comparison.iterrows():
    report_lines.append(f"{row['Model']:<20} | Acc: {row['Test Accuracy']:.4f} | F1: {row['Test F1 Score']:.4f}")

# ADD PER-CLASS ACCURACY TO REPORT
report_lines.extend([
    "",
    "PER-CLASS ACCURACY:",
    "-"*30
])

for model_name, results in final_results.items():
    report_lines.append(f"\n{model_name}:")
    if 'per_class_accuracy' in results:
        for class_name, acc in results['per_class_accuracy'].items():
            report_lines.append(f"  {class_name}: {acc:.4f}")
    report_lines.append(f"  Overall: {results['accuracy']:.4f}")

# ADD MISCLASSIFICATION ANALYSIS TO REPORT
report_lines.extend([
    "",
    "MISCLASSIFICATION ANALYSIS:",
    "-"*30
])

for model_name, results in final_results.items():
    if 'misclassified_df' in results:
        num_misclassified = len(results['misclassified_df'])
        total_test_samples = len(y_test)
        report_lines.append(f"\n{model_name}:")
        report_lines.append(f"  Total misclassified: {num_misclassified}/{total_test_samples} ({num_misclassified/total_test_samples*100:.1f}%)")
        
        if num_misclassified > 0:
            misclass_patterns = results['misclassified_df'].groupby(['True Label', 'Predicted Label']).size().sort_values(ascending=False)
            report_lines.append("  Most common misclassification patterns:")
            for (true_label, pred_label), count in misclass_patterns.head(3).items():
                report_lines.append(f"    {true_label} -> {pred_label}: {count} images")

report_lines.extend([
    "",
    f"Best performing model: {results_comparison.iloc[0]['Model']} (F1: {results_comparison.iloc[0]['Test F1 Score']:.4f})",
    "",
    "Files generated:",
    "- complete_checkpoint_optuna_optimized.pkl (full results)",
    "- best_hyperparameters.json (hyperparameters only)", 
    "- model_comparison_optimized.csv (performance comparison)",
    "- misclassified_images_[model].csv (misclassification analysis for each model)",
    "- optuna_optimization_history.png (optimization curves)",
    "- hyperparameter_importance.png (parameter importance)",
    "- cm_[model]_optimized.png (confusion matrices)",
    "- [model]_optimized_training_metrics.png (training curves)"
])

with open("optimization_report.txt", "w") as f:
    f.write("\n".join(report_lines))

# =============================================================================
# CREATE CONSOLIDATED MISCLASSIFICATION REPORT
# =============================================================================

print("\n" + "="*80)
print("CREATING CONSOLIDATED MISCLASSIFICATION REPORT")
print("="*80)

# Combine all misclassification data into one comprehensive report
all_misclassifications = []
for model_name, results in final_results.items():
    if 'misclassified_df' in results and len(results['misclassified_df']) > 0:
        df_copy = results['misclassified_df'].copy()
        df_copy['Model'] = model_name
        all_misclassifications.append(df_copy)

if all_misclassifications:
    consolidated_misclass_df = pd.concat(all_misclassifications, ignore_index=True)
    
    # Reorder columns to put Model first
    cols = ['Model'] + [col for col in consolidated_misclass_df.columns if col != 'Model']
    consolidated_misclass_df = consolidated_misclass_df[cols]
    
    # Save consolidated report
    consolidated_misclass_df.to_csv("consolidated_misclassifications_all_models.csv", index=False)
    print("Saved consolidated misclassification report to: consolidated_misclassifications_all_models.csv")
    
    # Create summary statistics
    print("\nMisclassification Summary Statistics:")
    print("="*40)
    
    # By model
    model_misclass_counts = consolidated_misclass_df['Model'].value_counts()
    print("\nMisclassifications by model:")
    for model, count in model_misclass_counts.items():
        percentage = (count / len(y_test)) * 100
        print(f"  {model}: {count} ({percentage:.1f}%)")
    
    # Most problematic images (misclassified by multiple models)
    image_misclass_counts = consolidated_misclass_df['Image Name'].value_counts()
    problematic_images = image_misclass_counts[image_misclass_counts > 1]
    
    if len(problematic_images) > 0:
        print(f"\nImages misclassified by multiple models ({len(problematic_images)} images):")
        for image, count in problematic_images.head(10).items():
            print(f"  {image}: misclassified by {count} models")
    
    # Most common true->predicted pairs across all models
    all_patterns = consolidated_misclass_df.groupby(['True Label', 'Predicted Label']).size().sort_values(ascending=False)
    print(f"\nMost common misclassification patterns across all models:")
    for (true_label, pred_label), count in all_patterns.head(5).items():
        print(f"  {true_label} -> {pred_label}: {count} instances")

print(f"\n✅ Hyperparameter optimization completed!")
print(f"✅ Best performing model: {results_comparison.iloc[0]['Model']} (F1: {results_comparison.iloc[0]['Test F1 Score']:.4f})")
print(f"✅ All results saved to 'complete_checkpoint_optuna_optimized.pkl'")
print(f"✅ Optimization report saved to 'optimization_report.txt'")
print(f"✅ Best hyperparameters saved to 'best_hyperparameters.json'")
print(f"✅ Individual misclassification reports saved as 'misclassified_images_[model].csv'")
print(f"✅ Consolidated misclassification report saved as 'consolidated_misclassifications_all_models.csv'")

# Clear GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nOptuna-optimized pipeline with misclassification analysis completed!")