import numpy as np
import soundfile as sf

from app.audio import cosine_similarity, extract_features


def test_extract_features_returns_vector(tmp_path):
    sample_rate = 22050
    duration = 1.0
    time = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * time)
    audio_path = tmp_path / "tone.wav"
    sf.write(audio_path, signal, sample_rate)

    vector = extract_features(audio_path)

    assert vector.shape == (20,)
    assert np.isfinite(vector).all()


def test_cosine_similarity():
    vector_a = np.array([1.0, 0.0, 0.0])
    vector_b = np.array([1.0, 0.0, 0.0])
    vector_c = np.array([0.0, 1.0, 0.0])

    assert cosine_similarity(vector_a, vector_b) == 1.0
    assert cosine_similarity(vector_a, vector_c) == 0.0
