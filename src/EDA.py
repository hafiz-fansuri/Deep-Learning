import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display
import seaborn as sns

# ============================================================
# PATHS
# ============================================================
CSV_PATH = r"C:\Users\fansuri\Documents\pro\DEEP LEARNING\birdclef-2026\train.csv"
AUDIO_PATH = r"C:\Users\fansuri\Documents\pro\DEEP LEARNING\birdclef-2026\train_audio"

OUT_DIR = "eda_outputsFS"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
MAX_PER_CLASS = 200
GLOBAL_SAMPLE = 1000
MIN_DURATION = 0.5  # filter very short clips

# ============================================================
# LOAD DATA
# ============================================================
print("\n[1] Loading dataset...")
df = pd.read_csv(CSV_PATH)

print("Total samples:", len(df))
classes = df['class_name'].unique()

def get_path(f):
    return os.path.join(AUDIO_PATH, f)

# ============================================================
# GLOBAL DISTRIBUTION
# ============================================================
print("\n[2] Global distribution...")

class_counts = df['class_name'].value_counts()

plt.figure(figsize=(8,5))
class_counts.plot(kind='bar')
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_distribution.png"))
plt.close()

plt.figure(figsize=(8,5))
class_counts.plot(kind='bar')
plt.yscale('log')
plt.title("Class Distribution (Log Scale)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_distribution_log.png"))
plt.close()

# ============================================================
# GLOBAL DURATION
# ============================================================
print("\n[3] Global duration...")

global_durations = []

sample_df = df.sample(min(GLOBAL_SAMPLE, len(df)), random_state=42)

for _, row in sample_df.iterrows():
    path = get_path(row['filename'])
    try:
        y, sr = librosa.load(path, sr=None)
        duration = len(y)/sr

        if duration < MIN_DURATION:
            continue

        global_durations.append(duration)
    except:
        continue

plt.figure()
plt.hist(global_durations, bins=30)
plt.title("Global Duration Distribution")
plt.xlabel("Seconds")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "global_duration.png"))
plt.close()

print("Min duration:", np.min(global_durations))
print("Max duration:", np.max(global_durations))

# ============================================================
# ALL CLASSES IN ONE (HIST + KDE) 🔥
# ============================================================
print("\n[4] Combined class comparison...")

plt.figure(figsize=(10,6))

for class_name in classes:
    class_df = df[df['class_name'] == class_name]

    durations = []

    sample_df = class_df.sample(min(MAX_PER_CLASS, len(class_df)), random_state=42)

    for _, row in sample_df.iterrows():
        path = get_path(row['filename'])
        try:
            y, sr = librosa.load(path, sr=None)
            duration = len(y)/sr

            if duration < MIN_DURATION:
                continue

            durations.append(duration)
        except:
            continue

    if len(durations) > 0:
        plt.hist(durations, bins=30, alpha=0.5, label=class_name)

plt.legend()
plt.yscale("log")
plt.title("Duration Distribution by Class (Overlay)")
plt.xlabel("Seconds")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "duration_all_classes.png"))
plt.close()

# KDE version (cleaner)
plt.figure(figsize=(10,6))

for class_name in classes:
    class_df = df[df['class_name'] == class_name]

    durations = []

    sample_df = class_df.sample(min(MAX_PER_CLASS, len(class_df)), random_state=42)

    for _, row in sample_df.iterrows():
        path = get_path(row['filename'])
        try:
            y, sr = librosa.load(path, sr=None)
            duration = len(y)/sr

            if duration < MIN_DURATION:
                continue

            durations.append(duration)
        except:
            continue

    if len(durations) > 5:
        sns.kdeplot(durations, label=class_name, fill=True)

plt.legend()
plt.title("Duration Distribution by Class (KDE)")
plt.xlabel("Seconds")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "duration_kde_all_classes.png"))
plt.close()

# ============================================================
# PER-CLASS ANALYSIS
# ============================================================
print("\n[5] Per-class analysis...")

summary_list = []

for class_name in classes:

    print(f"Processing class: {class_name}")

    class_df = df[df['class_name'] == class_name]

    durations, zcr_list, centroid_list, rms_list = [], [], [], []

    sample_df = class_df.sample(min(MAX_PER_CLASS, len(class_df)), random_state=42)

    for _, row in sample_df.iterrows():
        path = get_path(row['filename'])

        try:
            y, sr = librosa.load(path, sr=None, duration=5)

            duration = len(y)/sr
            if duration < MIN_DURATION:
                continue

            n_fft = min(2048, len(y))

            durations.append(duration)
            zcr_list.append(np.mean(librosa.feature.zero_crossing_rate(y)))
            centroid_list.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            rms_list.append(np.mean(librosa.feature.rms(y=y)))

        except:
            continue

    def save_hist(data, title, filename):
        if len(data) == 0:
            return
        plt.figure()
        plt.hist(data, bins=20)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, filename))
        plt.close()

    save_hist(durations, f"{class_name} Duration", f"{class_name}_duration.png")
    save_hist(zcr_list, f"{class_name} ZCR", f"{class_name}_zcr.png")
    save_hist(centroid_list, f"{class_name} Centroid", f"{class_name}_centroid.png")
    save_hist(rms_list, f"{class_name} RMS", f"{class_name}_rms.png")

    # Correlation (safe)
    if len(durations) > 5:
        feature_df = pd.DataFrame({
            "duration": durations,
            "zcr": zcr_list,
            "centroid": centroid_list,
            "rms": rms_list
        })

        if not feature_df.isnull().values.all():
            plt.figure()
            sns.heatmap(feature_df.corr(), annot=True)
            plt.title(f"{class_name} Feature Correlation")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"{class_name}_correlation.png"))
            plt.close()

    summary_list.append({
        "class": class_name,
        "count": len(class_df),
        "duration_mean": np.mean(durations) if durations else 0,
        "duration_min": np.min(durations) if durations else 0,
        "duration_max": np.max(durations) if durations else 0,
        "zcr_mean": np.mean(zcr_list) if zcr_list else 0,
        "centroid_mean": np.mean(centroid_list) if centroid_list else 0,
        "rms_mean": np.mean(rms_list) if rms_list else 0
    })

# ============================================================
# SUMMARY CSV
# ============================================================
summary_df = pd.DataFrame(summary_list)
summary_df.to_csv(os.path.join(OUT_DIR, "classwise_summary.csv"), index=False)

print("\n=== CLASS-WISE SUMMARY ===")
print(summary_df)

# ============================================================
# SPECTROGRAMS 🔥
# ============================================================
print("\n[6] Generating spectrograms...")

for class_name in classes:
    try:
        sample = df[df['class_name'] == class_name].sample(1).iloc[0]
        path = get_path(sample['filename'])

        y, sr = librosa.load(path, sr=22050)

        n_fft = min(2048, len(y))

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft)
        mel_db = librosa.power_to_db(mel)

        plt.figure(figsize=(8,4))
        librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Mel Spectrogram - {class_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{class_name}_mel.png"))
        plt.close()

    except:
        continue

# Grid view
fig, axes = plt.subplots(1, len(classes), figsize=(15,4))

for i, class_name in enumerate(classes):
    try:
        sample = df[df['class_name'] == class_name].sample(1).iloc[0]
        path = get_path(sample['filename'])

        y, sr = librosa.load(path, sr=22050)

        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel)

        librosa.display.specshow(mel_db, sr=sr, ax=axes[i])
        axes[i].set_title(class_name)
        axes[i].set_axis_off()

    except:
        axes[i].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "all_classes_mel.png"))
plt.close()

print("\n✅ EDA COMPLETE. Outputs saved in:", OUT_DIR)