import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score

# =========================
# CONFIG
# =========================
CSV_PATH = r"C:\Users\fansuri\Documents\pro\DEEP LEARNING\birdclef-2026\train.csv"
AUDIO_DIR = r"C:\Users\fansuri\Documents\pro\DEEP LEARNING\birdclef-2026\train_audio"
CACHE_DIR = "spectrogram_cache_fixed"

os.makedirs(CACHE_DIR, exist_ok=True)

SR = 16000
DURATION = 5
TARGET_LEN = SR * DURATION

BATCH_SIZE = 8
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)

classes = ["Aves", "Mammalia", "Reptilia", "Amphibia", "Insecta"]
df = df[df["class_name"].isin(classes)]

# ❗ REMOVE EXTREME LOW CLASS
df = df[df["class_name"] != "Reptilia"]

# =========================
# BALANCED SAMPLING
# =========================
max_per_class = 500

df = df.groupby("class_name").apply(
    lambda x: x.sample(min(len(x), max_per_class), random_state=42)
).reset_index(drop=True)

print("\nClass distribution after balancing:")
print(df["class_name"].value_counts())

# =========================
# LABEL ENCODING
# =========================
present_classes = sorted(df["class_name"].unique())
label_map = {c: i for i, c in enumerate(present_classes)}

df["label"] = df["class_name"].map(label_map)
NUM_CLASSES = len(present_classes)

# =========================
# AUDIO PROCESSING
# =========================
def process_audio(path):
    try:
        y, _ = librosa.load(path, sr=SR)

        if len(y) > TARGET_LEN:
            start = np.random.randint(0, len(y) - TARGET_LEN)
            y = y[start:start + TARGET_LEN]
        else:
            y = np.pad(y, (0, TARGET_LEN - len(y)))

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=SR,
            n_mels=128
        )

        mel = librosa.power_to_db(mel)

        # normalize per sample
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        return mel.astype(np.float32)

    except:
        return None

# =========================
# CACHE
# =========================
print("\nCaching spectrograms...")

valid_data = []

for i, row in df.iterrows():
    src_path = os.path.join(AUDIO_DIR, row["filename"])

    if not os.path.exists(src_path):
        continue

    cache_name = row["filename"].replace("/", "_") + ".npy"
    cache_path = os.path.join(CACHE_DIR, cache_name)

    if os.path.exists(cache_path):
        valid_data.append((cache_path, row["label"]))
        continue

    mel = process_audio(src_path)

    if mel is not None:
        np.save(cache_path, mel)
        valid_data.append((cache_path, row["label"]))

    if i % 200 == 0:
        print(f"Processed {i}/{len(df)}")

print("Total usable samples:", len(valid_data))

# =========================
# STRATIFIED SPLIT
# =========================
labels = [l for _, l in valid_data]

train_data, val_data = train_test_split(
    valid_data,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("Train:", len(train_data), "Val:", len(val_data))

# =========================
# CLASS WEIGHTS
# =========================
train_labels = [l for _, l in train_data]

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# =========================
# DATASET
# =========================
class CachedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]

        mel = np.load(path)

        mel = np.expand_dims(mel, 0)

        return (
            torch.tensor(mel, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )

train_loader = DataLoader(
    CachedDataset(train_data),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    CachedDataset(val_data),
    batch_size=BATCH_SIZE
)

# =========================
# MODEL (slightly improved)
# =========================
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = CNN(NUM_CLASSES).to(DEVICE)

# =========================
# TRAINING
# =========================
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("\nTraining...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

# =========================
# EVALUATION
# =========================
print("\nEvaluation...\n")

model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(DEVICE)
        out = model(x)
        preds = torch.argmax(out, dim=1)

        y_true.extend(y.numpy())
        y_pred.extend(preds.cpu().numpy())

print("Accuracy:", accuracy_score(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=present_classes,
    zero_division=0
))