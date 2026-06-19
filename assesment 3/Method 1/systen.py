"""
synthetic_audio_gen.py
Generates synthetic training audio for BirdCLEF 2026 missing species.
Strategy:
  - For sonotype variants (47158sonXX): pitch-shift + time-stretch real recordings of
    the base species (47158) to create acoustically distinct variants.
  - For pure-numeric missing labels: generate from Xeno-canto downloads if available,
    else create broadband noise bursts shaped to match typical bird call envelopes.
"""

import os, glob, random, numpy as np, soundfile as sf, librosa
from pathlib import Path

BASE_DIR    = r"C:\Users\fansuri\Documents\pro\DEEP LEARNING\birdclef-2026"
TRAIN_AUDIO = rf"{BASE_DIR}\train_audio"
OUT_DIR     = rf"{BASE_DIR}\synthetic_audio"
SR          = 32_000
SEG_SEC     = 5
N_SYNTH     = 20          # synthetic files per missing class
SEED        = 42
rng         = np.random.default_rng(SEED)

MISSING = [
    '116570', '1491113', '1595929', '25073', '516975', '74580',
    *[f'47158son{i:02d}' for i in range(1, 26)]
]

# ── helpers ────────────────────────────────────────────────────────────────

def load_random_segment(species_dir, sr=SR, duration=SEG_SEC):
    oggs = glob.glob(rf"{species_dir}\*.ogg")
    if not oggs:
        return None
    path = random.choice(oggs)
    try:
        y, _ = librosa.load(path, sr=sr, duration=duration)
        if len(y) < sr:
            return None
        if len(y) < duration * sr:
            y = np.pad(y, (0, duration * sr - len(y)))
        return y.astype(np.float32)
    except:
        return None

def augment_segment(y, sr=SR, pitch_shift=0.0, time_stretch=1.0, noise_level=0.002):
    if time_stretch != 1.0:
        y = librosa.effects.time_stretch(y, rate=time_stretch)
    if pitch_shift != 0.0:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
    # trim / pad back to SEG_SEC
    target = SEG_SEC * sr
    y = y[:target] if len(y) > target else np.pad(y, (0, max(0, target - len(y))))
    # add very light pink noise
    noise = rng.standard_normal(len(y)).astype(np.float32) * noise_level
    y = np.clip(y + noise, -1.0, 1.0)
    return y

def make_noise_burst(sr=SR, duration=SEG_SEC):
    """Fallback: amplitude-modulated bandpass noise resembling a bird call."""
    t    = np.linspace(0, duration, duration * sr)
    # random center freq between 1–8 kHz
    fc   = rng.uniform(1000, 8000)
    bw   = rng.uniform(500, 2000)
    lo   = max(50, fc - bw/2) / (sr/2)
    hi   = min(0.99, (fc + bw/2) / (sr/2))
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [lo, hi], btype='band')
    noise = rng.standard_normal(len(t)).astype(np.float32)
    y    = filtfilt(b, a, noise).astype(np.float32)
    # AM envelope: random bursts
    env  = np.zeros_like(t)
    for _ in range(rng.integers(3, 8)):
        start = rng.uniform(0, duration - 0.3)
        width = rng.uniform(0.1, 0.4)
        env  += np.exp(-0.5 * ((t - start) / (width/3))**2)
    env = (env / (env.max() + 1e-8)).astype(np.float32)
    return np.clip(y * env * 0.5, -1.0, 1.0)

# ── main ───────────────────────────────────────────────────────────────────

for label in MISSING:
    out_path = Path(OUT_DIR) / label
    out_path.mkdir(parents=True, exist_ok=True)

    # Determine seed source
    if 'son' in label:
        base_id  = '47158'          # base species for all sonotypes
        son_idx  = int(label.replace('47158son', ''))
        # spread pitch shifts evenly across 25 sonotypes: -6 to +6 semitones
        pitch    = -6.0 + (son_idx - 1) * (12.0 / 24.0)
        stretch  = rng.uniform(0.88, 1.12)
        src_dir  = rf"{TRAIN_AUDIO}\{base_id}"
    else:
        base_id  = label
        pitch    = rng.uniform(-2, 2)
        stretch  = rng.uniform(0.92, 1.08)
        src_dir  = rf"{TRAIN_AUDIO}\{base_id}"

    print(f"\n{label}  (pitch={pitch:+.1f} st, stretch={stretch:.2f})")

    for i in range(N_SYNTH):
        y = load_random_segment(src_dir)
        if y is None:
            print(f"  No source audio for {base_id} — using noise burst")
            y = make_noise_burst()
        else:
            y = augment_segment(y, pitch_shift=pitch, time_stretch=stretch,
                                noise_level=rng.uniform(0.0005, 0.003))

        fname = out_path / f"{label}_synth_{i:03d}.ogg"
        sf.write(str(fname), y, SR, format='OGG', subtype='VORBIS')

    print(f"  → {N_SYNTH} files written to {out_path}")

print("\nDone. Copy synthetic_audio/* into train_audio/ and re-run embedding extraction.")