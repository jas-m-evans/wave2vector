from __future__ import annotations

from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np


def extract_features(file_path: str | Path, *, sample_rate: int = 22050) -> np.ndarray:
    audio, sr = librosa.load(path=str(file_path), sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    return mfcc.mean(axis=1)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = (np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def plot_features(vector: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 2))
    plt.plot(vector, marker="o")
    plt.title("Audio Feature Vector")
    plt.xlabel("Feature Index")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
