import pathlib

import numpy as np
import pytest
import soundfile as sf

from pkg.audio_file_client.adapters.librosa_client import LibrosaClient
from pkg.audio_file_client.core import domain

TMP_PATH = pathlib.Path("/tmp")
SR = 16000
DURATION = 4
NUMBER_OF_MEL_BANDS = 128
HOP_LENGTH = 512


def test_the_test():
    assert 1 == 1


@pytest.fixture
def create_dummy_audio_file() -> pathlib.Path:
    t = np.linspace(0, DURATION, int(SR * DURATION))
    y = 0.5 * np.sin(2 * np.pi * 440 * t)  # Generate a 440 Hz tone
    test_file = TMP_PATH / "test_audio.wav"
    sf.write(test_file, y, SR)
    return test_file


def test_get_duration(create_dummy_audio_file):
    librosa_client = LibrosaClient()
    duration = librosa_client.get_duration(create_dummy_audio_file, SR)

    assert duration == DURATION


def test_crop(create_dummy_audio_file):
    librosa_client = LibrosaClient()
    cropped_amount = 2
    cropped_file = librosa_client.crop(
        create_dummy_audio_file, start_seconds=0, end_seconds=DURATION - cropped_amount
    )
    duration = librosa_client.get_duration(cropped_file, HOP_LENGTH)
    assert duration == DURATION - cropped_amount


def test_pad(create_dummy_audio_file):
    librosa_client = LibrosaClient()
    padded_amount = 2
    padded_file = librosa_client.pad(
        create_dummy_audio_file, duration=DURATION + padded_amount
    )
    duration = librosa_client.get_duration(padded_file, HOP_LENGTH)
    assert duration == DURATION + padded_amount


def test_extract_mel_spectrogram(create_dummy_audio_file):
    librosa_client = LibrosaClient()
    mel_spectrogram = librosa_client.extract_mel_spectrogram(
        create_dummy_audio_file,
        pre_processing_settings=domain.MelSpectrogramPreprocessingSettings(
            duration_seconds=DURATION,
            sampling_rate_hz=SR,
            number_of_mel_bands=NUMBER_OF_MEL_BANDS,
            hop_length=HOP_LENGTH,
        ),
    )
    assert mel_spectrogram.shape == (128, 126)
    assert isinstance(mel_spectrogram, np.ndarray)
