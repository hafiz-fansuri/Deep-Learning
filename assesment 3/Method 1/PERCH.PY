import kagglehub
import tensorflow as tf
import librosa
import numpy as np
import pandas as pd
import glob
import re
import os
import time
import pickle
from concurrent.futures import ThreadPoolExecutor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, top_k_accuracy_score
import warnings
warnings.filterwarnings("ignore")

tf.experimental.numpy.experimental_enable_numpy_behavior()

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
BASE_DIR        = r"C:\Users\fansuri\Documents\pro\DEEP LEARNING\birdclef-2026"
TRAIN_AUDIO     = rf"{BASE_DIR}\train_audio"
SOUNDSCAPES_DIR = rf"{BASE_DIR}\train_soundscapes"
OUT_DIR         = rf"{BASE_DIR}\outputs"
os.makedirs(OUT_DIR, exist_ok=True)

SR                    = 32_000
SEGMENT_SEC           = 5
LOAD_WORKERS          = 4
INFERENCE_BATCH       = 24
MAX_FILES_PER_SPECIES = 30



# ══════════════════════════════════════════════════════════════
# PERFORMANCE TRACKER
# ══════════════════════════════════════════════════════════════
class PerfTracker:
    def __init__(self):
        self.timings = {}
        self.counts  = {}
        self._starts = {}

    def start(self, key):
        self._starts[key] = time.time()

    def end(self, key, count=None):
        elapsed = time.time() - self._starts[key]
        self.timings[key] = self.timings.get(key, 0) + elapsed
        if count:
            self.counts[key] = self.counts.get(key, 0) + count
        return elapsed

    def report(self):
        print("\n" + "═"*55)
        print("  PERFORMANCE REPORT")
        print("═"*55)
        for key, t in self.timings.items():
            count = self.counts.get(key, "")
            rate  = f"  ({self.counts[key]/t:.1f}/s)" if key in self.counts and t > 0 else ""
            print(f"  {key:<35} {t:>8.1f}s  {count}{rate}")
        print("═"*55 + "\n")

perf = PerfTracker()

# ══════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════
print("Loading Perch model...")
perf.start("model_load")
path = kagglehub.model_download("google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu")
birdclassifier = tf.saved_model.load(path)
perf.end("model_load")
print(f"Model loaded from: {path}")

# ══════════════════════════════════════════════════════════════
# LABEL MAPPING
# ══════════════════════════════════════════════════════════════
bc_labels = (
    pd.read_csv(f"{path}/assets/labels.csv")
      .reset_index()
      .rename({"inat2024_fsd50k": "scientific_name", "index": "bc_index"}, axis=1)
      .set_index("scientific_name")
)
taxonomy   = pd.read_csv(rf"{BASE_DIR}\taxonomy.csv")
sample_sub = pd.read_csv(rf"{BASE_DIR}\sample_submission.csv")

mapping = taxonomy.join(bc_labels, on="scientific_name", how="left")
mapping.bc_index = mapping.bc_index.fillna(len(bc_labels)).astype(int)
mapping = mapping[["primary_label", "bc_index"]].set_index("primary_label")

primary_labels         = sample_sub.columns[1:].tolist()
birdclassifier_indices = [int(mapping.loc[pl].iloc[0]) for pl in primary_labels]

known_species   = [pl for pl in primary_labels if mapping.loc[pl].iloc[0] != len(bc_labels)]
unknown_species = [pl for pl in primary_labels if mapping.loc[pl].iloc[0] == len(bc_labels)]
print(f"Perch knows:   {len(known_species)}/234 species")
print(f"Missing:       {len(unknown_species)}/234 species → covered by MLP")
print(f"Missing list:  {unknown_species}\n")

# ══════════════════════════════════════════════════════════════
# AUDIO UTILS
# ══════════════════════════════════════════════════════════════
def load_segments(fpath):
    try:
        duration = librosa.get_duration(path=fpath)
        if duration < 1.0:
            return []
        segments = []
        for start in range(0, int(duration), SEGMENT_SEC):
            chunk, _ = librosa.load(fpath, sr=SR, offset=start, duration=float(SEGMENT_SEC))
            if len(chunk) < SEGMENT_SEC * SR:
                chunk = np.pad(chunk, (0, SEGMENT_SEC * SR - len(chunk)))
            if np.max(np.abs(chunk)) > 1e-6:
                segments.append(chunk)
        return segments
    except Exception as e:
        print(f"    [SKIP] {fpath}: {e}")
        return []

def extract_embeddings(chunks):
    batch   = np.stack(chunks).astype(np.float32)
    outputs = birdclassifier.signatures["serving_default"](inputs=batch)
    if "embedding" in outputs:
        return outputs["embedding"].numpy()
    return outputs["label"].numpy()

# ══════════════════════════════════════════════════════════════
# STEP 1 — EXTRACT EMBEDDINGS (cached)
# ══════════════════════════════════════════════════════════════
EMBEDDINGS_CACHE = rf"{OUT_DIR}\embeddings.pkl"

if os.path.exists(EMBEDDINGS_CACHE):
    print(f"Loading cached embeddings from {EMBEDDINGS_CACHE}")
    with open(EMBEDDINGS_CACHE, "rb") as f:
        emb_data = pickle.load(f)
    X_train_emb = emb_data["X"]
    y_train     = emb_data["y"]
    print(f"Loaded {len(X_train_emb)} embeddings across {len(set(y_train))} species\n")

else:
    print("Extracting embeddings from train_audio...")
    perf.start("embedding_extraction")

    X_train_emb = []
    y_train     = []
    total_files = 0
    skipped     = 0

    species_dirs = sorted(glob.glob(rf"{TRAIN_AUDIO}\*"))
    print(f"Found {len(species_dirs)} species folders\n")

    for sp_dir in species_dirs:
        species = os.path.basename(sp_dir)
        if species not in primary_labels:
            continue

        oggs = glob.glob(rf"{sp_dir}\*.ogg")[:MAX_FILES_PER_SPECIES]
        sp_embeddings = []

        with ThreadPoolExecutor(max_workers=LOAD_WORKERS) as executor:
            all_segments = list(executor.map(load_segments, oggs))

        for fpath, segments in zip(oggs, all_segments):
            if not segments:
                skipped += 1
                continue
            for i in range(0, len(segments), INFERENCE_BATCH):
                batch_chunks = segments[i:i+INFERENCE_BATCH]
                try:
                    embs = extract_embeddings(batch_chunks)
                    sp_embeddings.extend(embs)
                    y_train.extend([species] * len(embs))
                except Exception as e:
                    print(f"    [WARN] Embedding failed: {e}")
            total_files += 1

        X_train_emb.extend(sp_embeddings)
        print(f"  {species:<20} {len(sp_embeddings):>5} embeddings from {len(oggs)} files")

    X_train_emb = np.array(X_train_emb)
    y_train     = np.array(y_train)

    elapsed = perf.end("embedding_extraction", count=total_files)
    print(f"\nExtracted {len(X_train_emb)} embeddings from {total_files} files "
          f"({skipped} skipped) in {elapsed:.1f}s")

    with open(EMBEDDINGS_CACHE, "wb") as f:
        pickle.dump({"X": X_train_emb, "y": y_train}, f)
    print(f"Embeddings cached → {EMBEDDINGS_CACHE}")

# ══════════════════════════════════════════════════════════════
# STEP 2 — TRAIN MLP (cached)
# ══════════════════════════════════════════════════════════════
MLP_CACHE = rf"{OUT_DIR}\mlp_model.pkl"

if os.path.exists(MLP_CACHE):
    print(f"\nLoading cached MLP from {MLP_CACHE}")
    with open(MLP_CACHE, "rb") as f:
        mlp_data = pickle.load(f)
    mlp            = mlp_data["mlp"]
    le             = mlp_data["le"]
    primary_labels = mlp_data["primary_labels"]
    known_species  = mlp_data["known_species"]
    unknown_species= mlp_data["unknown_species"]
    print("MLP loaded.\n")

else:
    print("\nTraining MLP classifier...")
    perf.start("mlp_training")

    le        = LabelEncoder()
    y_encoded = le.fit_transform(y_train)

    # all 234 competition labels known to the encoder
    all_classes = np.array(primary_labels)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_emb, y_encoded,
        test_size=0.15,
        random_state=42,
        stratify=y_encoded
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation="relu",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        verbose=True,
        random_state=42
    )
    mlp.fit(X_tr, y_tr)
    elapsed = perf.end("mlp_training")

    # ── Evaluation ────────────────────────────────────────────
    print(f"\nMLP trained in {elapsed:.1f}s")
    print("\n── Validation Performance ──────────────────────────")

    y_pred       = mlp.predict(X_val)
    y_pred_proba = mlp.predict_proba(X_val)

    # FIX: pass labels= so sklearn knows the full class list
    unique_val_classes = np.unique(y_val)
    top1 = (y_pred == y_val).mean()
    top5 = top_k_accuracy_score(
        y_val,
        y_pred_proba,
        k=5,
        labels=np.arange(len(le.classes_))   # ← the fix
    )

    print(f"  Top-1 Accuracy:  {top1*100:.2f}%")
    print(f"  Top-5 Accuracy:  {top5*100:.2f}%")
    print(f"  Classes in MLP:  {len(le.classes_)}")
    print(f"  Classes in val:  {len(unique_val_classes)}")

    # per-species report
    label_names  = le.inverse_transform(np.arange(len(le.classes_)))
    report       = classification_report(
                       y_val, y_pred,
                       labels=np.arange(len(le.classes_)),   # ← fix
                       target_names=label_names,
                       output_dict=True,
                       zero_division=0
                   )
    report_df    = pd.DataFrame(report).T
    report_path  = rf"{OUT_DIR}\mlp_classification_report.csv"
    report_df.to_csv(report_path)
    print(f"\n  Per-species report → {report_path}")

    # 10 worst species
    species_rows = report_df[report_df.index.isin(primary_labels)]
    worst10      = species_rows["f1-score"].sort_values().head(10)
    print(f"\n  10 hardest species (lowest F1):")
    print(worst10.to_string())

    # ── Save MLP + all metadata needed for inference ──────────
    with open(MLP_CACHE, "wb") as f:
        pickle.dump({
            "mlp":             mlp,
            "le":              le,
            "primary_labels":  primary_labels,
            "known_species":   known_species,
            "unknown_species": unknown_species,
            "birdclassifier_indices": birdclassifier_indices,
            "top1_acc":        top1,
            "top5_acc":        top5,
        }, f)
    print(f"\nMLP saved → {MLP_CACHE}")

    # ── Also export a standalone zip for Kaggle upload ────────
    import shutil
    kaggle_export = rf"{OUT_DIR}\kaggle_model"
    os.makedirs(kaggle_export, exist_ok=True)
    shutil.copy(MLP_CACHE, rf"{kaggle_export}\mlp_model.pkl")

    # save a small metadata json for easy loading on Kaggle
    import json
    meta = {
        "primary_labels":         primary_labels,
        "known_species":          known_species,
        "unknown_species":        unknown_species,
        "birdclassifier_indices": birdclassifier_indices,
        "top1_acc":               round(top1, 4),
        "top5_acc":               round(top5, 4),
        "n_classes":              len(le.classes_),
        "mlp_layers":             list(mlp.hidden_layer_sizes),
    }
    with open(rf"{kaggle_export}\meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    shutil.make_archive(rf"{OUT_DIR}\kaggle_model", "zip", kaggle_export)
    print(f"Kaggle export zip → {OUT_DIR}\\kaggle_model.zip")

# ══════════════════════════════════════════════════════════════
# STEP 3 — SOUNDSCAPE INFERENCE
# ══════════════════════════════════════════════════════════════
print("\nRunning inference on soundscapes...")
perf.start("soundscape_inference")

oggs = glob.glob(rf"{SOUNDSCAPES_DIR}\*.ogg")
if not oggs:
    oggs = glob.glob(rf"{BASE_DIR}\test_soundscapes\*.ogg")
print(f"Found {len(oggs)} soundscape files")

out_path    = rf"{OUT_DIR}\submission.csv"
done_ss_ids = set()
if os.path.exists(out_path):
    existing    = pd.read_csv(out_path)
    done_ss_ids = set(existing["row_id"].str.rsplit("_", n=1).str[0])
    print(f"Resuming — {len(done_ss_ids)} already done")

oggs = [f for f in oggs
        if re.search(r"[\\/]([^\\/]+)\.ogg$", f).group(1) not in done_ss_ids]
print(f"Remaining: {len(oggs)} files\n")

mlp_classes = list(le.classes_)
all_ids     = []
all_results = []
skipped     = 0

def process_soundscape(fpath):
    ss_id    = re.search(r"[\\/]([^\\/]+)\.ogg$", fpath).group(1)
    segments = load_segments(fpath)
    if not segments:
        return ss_id, [], None

    row_ids = [f"{ss_id}_{(i+1)*SEGMENT_SEC}" for i in range(len(segments))]

    try:
        batch     = np.stack(segments).astype(np.float32)
        perch_out = birdclassifier.signatures["serving_default"](inputs=batch)

        # perch label scores → shape (N, 234)
        perch_labels = tf.pad(perch_out["label"], tf.constant([[0, 0], [0, 1]]))
        perch_scores = perch_labels[:, birdclassifier_indices].numpy()

        # embeddings → MLP proba
        embs         = perch_out["embedding"].numpy() if "embedding" in perch_out \
                       else perch_out["label"].numpy()
        mlp_proba    = mlp.predict_proba(embs)          # (N, n_mlp_classes)

        # map MLP columns → primary_labels order
        mlp_scores = np.zeros((len(segments), len(primary_labels)), dtype=np.float32)
        for i, sp in enumerate(mlp_classes):
            if sp in primary_labels:
                idx = primary_labels.index(sp)
                mlp_scores[:, idx] = mlp_proba[:, i]

        # blend: 0.6 perch + 0.4 mlp for known, 1.0 mlp for unknown
        blended = np.zeros_like(perch_scores)
        for j, pl in enumerate(primary_labels):
            if pl in known_species:
                blended[:, j] = 0.6 * perch_scores[:, j] + 0.4 * mlp_scores[:, j]
            else:
                blended[:, j] = mlp_scores[:, j]

        return ss_id, row_ids, blended

    except Exception as e:
        print(f"  [SKIP] {ss_id}: {e}")
        return ss_id, [], None

for i, fpath in enumerate(oggs):
    ss_id, row_ids, result = process_soundscape(fpath)
    print(f"  [{i+1}/{len(oggs)}] {ss_id} — {len(row_ids)} segments")

    if result is None:
        skipped += 1
        continue

    all_ids     += row_ids
    all_results.append(result)

    # checkpoint every 50 files
    if (i + 1) % 50 == 0 and all_results:
        ckpt = pd.DataFrame(np.concatenate(all_results), columns=primary_labels)
        ckpt.insert(0, "row_id", all_ids)
        if os.path.exists(out_path):
            ckpt = pd.concat([pd.read_csv(out_path), ckpt], ignore_index=True)
        ckpt.to_csv(out_path, index=False)
        print(f"  ✓ Checkpoint saved ({len(all_ids)} rows so far)")
        all_ids, all_results = [], []

elapsed = perf.end("soundscape_inference", count=len(oggs))

# ── Final save ────────────────────────────────────────────────
if all_results:
    submission = pd.DataFrame(np.concatenate(all_results), columns=primary_labels)
    submission.insert(0, "row_id", all_ids)
    if os.path.exists(out_path):
        submission = pd.concat([pd.read_csv(out_path), submission], ignore_index=True)
    submission.to_csv(out_path, index=False)

print(f"\nSkipped: {skipped} | Done → {out_path}")
if os.path.exists(out_path):
    final = pd.read_csv(out_path)
    print(f"Final submission shape: {final.shape}")

perf.report()