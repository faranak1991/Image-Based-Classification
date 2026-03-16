import io
import os
import pickle
import tempfile
import warnings

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models

warnings.filterwarnings("ignore")
device = torch.device("cpu")


# ========================= MODEL CLASSES =========================

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
        return self.classifier(x)


class DenseNet121(nn.Module):
    def __init__(self, num_classes=5, dropout=0.4, fc_hidden=256):
        super().__init__()
        backbone = models.densenet121(pretrained=False)

        old_conv = backbone.features.conv0
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
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
        return self.classifier(x)


class ResNet18(nn.Module):
    def __init__(self, num_classes=5, dropout=0.4, fc_hidden=256, pretrained=True):
        super().__init__()
        backbone = models.resnet18(pretrained=False)

        old = backbone.conv1
        new = nn.Conv2d(
            1, old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False
        )
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
        return self.classifier(x)


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


# ========================= CUSTOM UNPICKLER =========================

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


# ========================= HELPERS =========================

def load_and_preprocess_image(img_path, img_size=(224, 224)):
    pil_img = Image.open(img_path)
    gray = pil_img.convert("L")
    gray = gray.resize(img_size)
    arr = np.array(gray, dtype=np.uint8)
    return arr.astype(np.float32) / 255.0


def prepare_model(model):
    if hasattr(model, "module"):
        model = model.module
    model = model.to(device)
    model.eval()
    return model


def get_nn_probabilities(model, X_data, batch_size=8):
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
                         optimized_weights, class_names, img_size):
    img = load_and_preprocess_image(img_path, img_size)
    X_single = np.array([img], dtype=np.float32)

    model_probs = {}
    for model_name, model in available_models.items():
        probs = get_nn_probabilities(model, X_single)
        model_probs[model_name] = probs

    member_probs = []
    available_members = []

    for k in ensemble_members:
        if k in model_probs:
            member_probs.append(model_probs[k])
            available_members.append(k)

    if not member_probs:
        raise RuntimeError("No ensemble members found.")

    available_indices = [ensemble_members.index(m) for m in available_members]
    weights = optimized_weights[available_indices]
    weights = weights / weights.sum()

    stacked = np.stack(member_probs, axis=0)
    avg_probs = np.tensordot(weights, stacked, axes=(0, 0))

    row_sums = avg_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    avg_probs = avg_probs / row_sums

    y_pred = np.argmax(avg_probs, axis=1)[0]
    predicted_label = class_names[y_pred]
    class_probabilities = {
        class_names[i]: float(avg_probs[0, i]) for i in range(len(class_names))
    }

    return predicted_label, class_probabilities


@st.cache_resource
def load_checkpoint_and_models():
    with open("complete_checkpoint_optuna_optimized.pkl", "rb") as f:
        checkpoint = CPU_Unpickler(f).load()

    config = checkpoint.get("config", {})
    class_names = config.get(
        "CLASS_NAMES",
        ["ADLs", "Aggregates", "Droplets", "Gels", "Solutions"]
    )
    display_name_map = {"Solutions": "Solution"}
    class_names = [display_name_map.get(n, n) for n in class_names]
    img_size = config.get("IMG_SIZE", (224, 224))

    final_results = checkpoint.get("final_results", {})
    optimized_weights = np.array([0.279, 0.15685, 0.564], dtype=float)
    ensemble_members = ["ResNet", "EfficientNet", "DenseNet"]

    available_models = {}
    for model_name in ensemble_members:
        if model_name in final_results and "model" in final_results[model_name]:
            model = final_results[model_name]["model"]
            if model is not None:
                available_models[model_name] = prepare_model(model)

    return class_names, img_size, optimized_weights, ensemble_members, available_models


def run_batch_prediction(uploaded_files):
    class_names, img_size, optimized_weights, ensemble_members, available_models = load_checkpoint_and_models()

    results = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            temp_path = tmp.name
            image.convert("L").save(temp_path)

        predicted_label, class_probs = predict_single_image(
            temp_path,
            available_models,
            ensemble_members,
            optimized_weights,
            class_names,
            img_size
        )

        try:
            os.remove(temp_path)
        except OSError:
            pass

        results.append({
            "filename": uploaded_file.name,
            "image": image.copy(),
            "predicted_label": predicted_label,
            "class_probs": class_probs
        })

    return results


def results_to_dataframe(results):
    rows = []
    for item in results:
        row = {
            "Image": item["filename"],
            "Predicted Label": item["predicted_label"]
        }
        for class_name, prob in item["class_probs"].items():
            row[class_name] = round(prob, 6)
        rows.append(row)
    return pd.DataFrame(rows)


# ========================= PAGE / STYLE =========================

st.set_page_config(page_title="Protein State Classification", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1350px;
        padding-top: 1.3rem;
        padding-bottom: 2rem;
    }
    .title-main {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }
    .subtitle {
        font-size: 1.15rem;
        color: #444;
        margin-bottom: 1rem;
    }
    .row-card {
        border: 1px solid #dddddd;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 20px;
        background: #ffffff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .pred-box {
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        padding: 16px;
        background: #fafafa;
    }
    .file-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }
    .pred-title {
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 0.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title-main">Condensate State Classification</div>', unsafe_allow_html=True)


uploaded_files = st.file_uploader(
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    accept_multiple_files=True
)

button_col1, button_col2 = st.columns([1, 1])
with button_col1:
    clear_clicked = st.button("Clear", use_container_width=True)
with button_col2:
    submit_clicked = st.button("Submit", use_container_width=True)

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None

if clear_clicked:
    st.session_state.batch_results = None
    st.rerun()

if submit_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one image.")
        st.session_state.batch_results = None
    else:
        with st.spinner("Running predictions..."):
            st.session_state.batch_results = run_batch_prediction(uploaded_files)

results = st.session_state.batch_results

if results:
    for item in results:
        left_col, right_col = st.columns([1, 1.15], gap="large")

        with left_col:
            st.markdown(f"### {item['filename']}")
            st.image(item["image"], use_column_width=True)

        with right_col:
            st.markdown(f"## Prediction: {item['predicted_label']}")

            sorted_probs = sorted(item["class_probs"].items(), key=lambda x: x[1], reverse=True)
            for class_name, prob in sorted_probs:
                st.write(f"**{class_name}** — {prob:.2%}")
                st.progress(float(prob))

        st.markdown("---")

         

    df = results_to_dataframe(results)
    st.markdown("---")
    st.subheader("Summary Table")
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download Results as CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="protein_state_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.info("Upload images and click Submit to see predictions.")