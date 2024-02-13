import pathlib
import queue
from abc import ABC, abstractmethod

from pkg.audio_file_client.core import domain


class Classifier(ABC):
    @abstractmethod
    def classify(
        self,
        path: pathlib.Path,
        mel_spectrogram_preprocessing_settings: domain.MelSpectrogramPreprocessingSettings,
    ) -> float:
        """
        Classify the audio file
        return the probability that the audio contains what the model is trained on
        """


class Recorder(ABC):
    @abstractmethod
    def record(self, recording_rate_khz: int, duration: float) -> pathlib.Path:
        """
        Record audio and save it to the path
        """

    def record_async(self, recording_rate_khz: int, duration: float) -> queue.Queue:
        """
        Record audio and save it to the path
        """


class Service(ABC):
    @abstractmethod
    def evaluate_from_microphone(
        self, recording_rate_khz: int, duration: float
    ) -> float:
        """
        Record audio and classify it
        return the probability that the audio contains a baby crying
        """
