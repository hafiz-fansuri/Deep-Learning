import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa

# ============================================================
# PATHS
# ============================================================
CSV_PATH = r"C:\Users\fansuri\Documents\pro\DEEP LEARNING\birdclef-2026\train.csv"
AUDIO_PATH = r"C:\Users\fansuri\Documents\pro\DEEP LEARNING\birdclef-2026\train_audio"

OUT_DIR = "eda_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(CSV_PATH)

# ============================================================
# GLOBAL CLASS DISTRIBUTION
# ============================================================
class_counts = df['class_name'].value_counts()

plt.figure(figsize=(8,5))
class_counts.plot(kind='bar')
plt.title("Global Class Distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_distribution.png"))
plt.close()

# ============================================================
# AUDIO PATH HELPER
# ============================================================
def get_path(f):
    return os.path.join(AUDIO_PATH, f)

# ============================================================
# STORE CLASS-WISE METRICS
# ============================================================
summary_list = []

# ============================================================
# LOOP THROUGH EACH CLASS
# ============================================================
for class_name in df['class_name'].unique():

    print(f"Processing class: {class_name}")

    class_df = df[df['class_name'] == class_name]

    # -------------------------
    # SAMPLE AUDIO FOR ANALYSIS
    # -------------------------
    durations = []
    zcr_list = []
    centroid_list = []
    rms_list = []

    sample_df = class_df.sample(min(200, len(class_df)), random_state=42)

    for _, row in sample_df.iterrows():
        path = get_path(row['filename'])
        try:
            y, sr = librosa.load(path, sr=None, duration=5)

            durations.append(len(y)/sr)
            zcr_list.append(np.mean(librosa.feature.zero_crossing_rate(y)))
            centroid_list.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            rms_list.append(np.mean(librosa.feature.rms(y=y)))

        except:
            continue

    # -------------------------
    # PLOT DURATION DISTRIBUTION
    # -------------------------
    plt.figure()
    plt.hist(durations, bins=20)
    plt.title(f"{class_name} - Duration Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{class_name}_duration.png"))
    plt.close()

    # -------------------------
    # FEATURE DISTRIBUTIONS
    # -------------------------
    plt.figure()
    plt.hist(zcr_list, bins=20)
    plt.title(f"{class_name} - Zero Crossing Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{class_name}_zcr.png"))
    plt.close()

    plt.figure()
    plt.hist(centroid_list, bins=20)
    plt.title(f"{class_name} - Spectral Centroid")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{class_name}_centroid.png"))
    plt.close()

    plt.figure()
    plt.hist(rms_list, bins=20)
    plt.title(f"{class_name} - RMS Energy")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{class_name}_rms.png"))
    plt.close()

    # -------------------------
    # SUMMARY STATS PER CLASS
    # -------------------------
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
# FINAL SUMMARY TABLE
# ============================================================
summary_df = pd.DataFrame(summary_list)

summary_df.to_csv(os.path.join(OUT_DIR, "classwise_summary.csv"), index=False)

print("\n=== CLASS-WISE SUMMARY ===")
print(summary_df)

# ============================================================
# IMBALANCE VISUALIZATION (LOG SCALE)
# ============================================================
plt.figure(figsize=(8,5))
class_counts.plot(kind='bar')
plt.yscale('log')
plt.title("Class Distribution (Log Scale)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_distribution_log.png"))
plt.close()

print("\nEDA completed. All outputs saved in:", OUT_DIR)