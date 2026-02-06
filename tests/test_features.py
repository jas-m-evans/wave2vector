import numpy as np
import soundfile as sf

from app.audio import extract_features


def test_mfcc_vector_length_is_40(tmp_path):
    sample_rate = 22050
    duration = 1.0
    time = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * time)
    audio_path = tmp_path / "tone.wav"
    sf.write(audio_path, signal, sample_rate)

    vector = extract_features(audio_path)

    assert vector.shape == (40,)


def test_extract_features_deterministic_on_synthetic_input(tmp_path):
    sample_rate = 22050
    duration = 1.0
    time = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 220 * time)
    audio_path = tmp_path / "tone.wav"
    sf.write(audio_path, signal, sample_rate)

    vector_first = extract_features(audio_path)
    vector_second = extract_features(audio_path)

    assert np.allclose(vector_first, vector_second)
