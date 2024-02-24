import pathlib
from abc import ABC, abstractmethod

import numpy as np

from cry_baby.pkg.audio_file_client.core import domain


class AudioFileClient(ABC):
    @abstractmethod
    def get_duration(
        self, path_to_audio_file: pathlib.Path, hop_length: int, sampling_rate_hz: int
    ) -> float:
        """
        Get the duration of the audio file in seconds
        """

    @abstractmethod
    def crop(
        self, path: pathlib.Path, start_seconds: float, end_seconds: float
    ) -> pathlib.Path:
        """
        Crop the audio file and return the path to the cropped audio file
        """

    @abstractmethod
    def pad(self, path: pathlib.Path, duration: float) -> pathlib.Path:
        """
        Pad the audio file and return the path to the padded audio file
        This assumes that the audio file is shorter than the duration
        """

    @abstractmethod
    def extract_mel_spectrogram(
        self,
        audio_file_path: pathlib.Path,
        pre_processing_settings: domain.MelSpectrogramPreprocessingSettings,
    ) -> np.ndarray:
        """
        Extract the mel spectrogram
        """

    @abstractmethod
    def check_audio_loudness(
        self, file_path: pathlib.Path, threshold_db: float, sampling_rate_hz: int
    ) -> bool:
        """
        Check if the audio file's loudness exceeds a given threshold in decibels.
        """
