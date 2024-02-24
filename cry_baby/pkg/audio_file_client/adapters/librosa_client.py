import pathlib
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from librosa.feature import melspectrogram

from cry_baby.pkg.audio_file_client.core import domain, ports
from cry_baby.pkg.audio_file_client.core.domain import (
    LoadError,
    UnexpectedDurationError,
)


class LibrosaClient(ports.AudioFileClient):
    cached_files: dict[pathlib.Path, tuple[np.ndarray, float]] = {}

    def extract_mel_spectrogram(
        self,
        audio_file_path: pathlib.Path,
        pre_processing_settings: domain.MelSpectrogramPreprocessingSettings,
    ) -> np.ndarray:
        """
        Its assumed the audio files passed into this function are already
        trimmed to the correct length
        """
        if not audio_file_path.exists() or not audio_file_path.is_file():
            raise FileNotFoundError

        # Check that the duration of the audio file is as long as the pre_processing_settings.duration_seconds,
        # within a margin of error
        if (
            duration := round(
                self.get_duration(
                    path_to_audio_file=audio_file_path,
                    hop_length=pre_processing_settings.hop_length,
                    sampling_rate_hz=pre_processing_settings.sampling_rate_hz,
                ),
                1,
            )
        ) != round(pre_processing_settings.duration_seconds, 1):
            raise UnexpectedDurationError(
                f"Audio file {audio_file_path} has duration {duration} seconds, "
                f"but the pre_processing_settings.duration_seconds is {pre_processing_settings.duration_seconds}"
            )

        y, sr = self._load(audio_file_path, pre_processing_settings.sampling_rate_hz)

        if sr != pre_processing_settings.sampling_rate_hz:
            raise ValueError(
                f"Audio file {audio_file_path} has sampling rate {sr}, "
                f"but the pre_processing_settings.sampling_rate_hz is {pre_processing_settings.sampling_rate_hz}"
                "they should be the same"
            )

        mel_spectrogram = melspectrogram(
            y=y,
            sr=sr,
            n_mels=pre_processing_settings.number_of_mel_bands,
            hop_length=pre_processing_settings.hop_length,
        )

        target_shape = _calc_target_shape(
            pre_processing_settings.sampling_rate_hz,
            pre_processing_settings.duration_seconds,
            pre_processing_settings.number_of_mel_bands,
            pre_processing_settings.hop_length,
        )

        if mel_spectrogram.shape != target_shape:
            mel_spectrogram = _resize_matrix(mel_spectrogram, target_shape)

        # Taking the logarithm of the Mel spectrogram is a common step because
        # human perception of sound intensity is logarithmic in nature
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

        # Normalization
        log_mel_spectrogram = (
            log_mel_spectrogram - np.mean(log_mel_spectrogram)
        ) / np.std(log_mel_spectrogram)

        return log_mel_spectrogram

    def get_duration(
        self, path_to_audio_file: pathlib.Path, hop_length: int, sampling_rate_hz: int
    ) -> float:
        """
        Get the duration of the audio file in seconds using librosa.
        """
        y, sr = self._load(path_to_audio_file, sampling_rate_hz)
        return librosa.get_duration(y=y, sr=sr, hop_length=hop_length)

    def crop(
        self, path: pathlib.Path, start_seconds: float, end_seconds: float
    ) -> pathlib.Path:
        """
        Crop the audio file using librosa and return the path to the cropped audio file.
        """
        y, sr = librosa.load(
            path, sr=None, offset=start_seconds, duration=end_seconds - start_seconds
        )
        cropped_file_path = path.with_suffix(".cropped" + path.suffix)
        sf.write(cropped_file_path, y, sr)
        return cropped_file_path

    def pad(self, path: pathlib.Path, duration: float) -> pathlib.Path:
        """
        Pad the audio file with silence to the specified duration and return the path to the padded audio file.
        This assumes that the audio file is shorter than the duration.
        """
        y, sr = librosa.load(path, sr=None)
        current_duration = librosa.get_duration(y=y, sr=sr)

        if current_duration < duration:
            # Calculate the length of the silence to add (in samples)
            silence_length = int((duration - current_duration) * sr)

            # Pad with zeros (silence)
            padded_audio = np.concatenate((y, np.zeros(silence_length)))

            padded_file_path = path.with_suffix(".padded" + path.suffix)
            sf.write(padded_file_path, padded_audio, sr)
            return padded_file_path
        else:
            # If the audio is already longer than the specified duration, return the original path
            return path

    def check_audio_loudness(
        self, file_path: pathlib.Path, threshold_db: float, sampling_rate_hz: int
    ) -> bool:
        """
        Check if the audio file's loudness exceeds a given threshold in decibels.
        """
        y, _ = self._load(file_path, sampling_rate_hz=sampling_rate_hz)
        rms_amplitude = librosa.feature.rms(y=y)[0]
        rms_db = librosa.amplitude_to_db(rms_amplitude, ref=0.00002)
        return any(rms_db > threshold_db)

    @staticmethod
    def _load(
        path: pathlib.Path, sampling_rate_hz: Optional[int]
    ) -> tuple[np.ndarray, float]:
        """
        Load the audio file using librosa.
        If it is not contained in our local in memory cache.
        """
        cache = LibrosaClient.cached_files.get(path)
        if cache:
            return cache
        try:
            y, sr = librosa.load(path, sr=sampling_rate_hz)
        except Exception as e:
            raise LoadError(f"Error loading audio file {path}: {e}")
        LibrosaClient.cached_files[path] = (y, sr)
        return y, sr


def _calc_target_shape(
    sampling_rate_hz: int,
    duration_seconds: int,
    number_of_mel_bands: int,
    hop_length: int,
) -> tuple[int, int]:
    number_of_samples = duration_seconds * sampling_rate_hz
    number_of_frames = int((number_of_samples + hop_length) / hop_length)
    return number_of_mel_bands, number_of_frames


def _resize_matrix(matrix: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if matrix.shape[1] < target_shape[1]:
        padding = np.zeros(
            (
                matrix.shape[0],
                target_shape[1] - matrix.shape[1],
            )
        )
        return np.hstack((matrix, padding))
    elif matrix.shape[1] > target_shape[1]:
        matrix = matrix[:, : target_shape[1]]
    return matrix
