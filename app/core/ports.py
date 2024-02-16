import pathlib
from abc import ABC, abstractmethod
from typing import Optional

from pkg.audio_file_client.core.domain import MelSpectrogramPreprocessingSettings


class Classifier(ABC):
    mel_spectrogram_preprocessing_settings: MelSpectrogramPreprocessingSettings

    @abstractmethod
    def classify(
        self,
        path_to_audio_file: pathlib.Path,
    ) -> float:
        """
        Classify the audio file
        return the probability that the audio contains what the model is trained on
        """


class Recorder(ABC):
    @abstractmethod
    def record(self) -> pathlib.Path:
        """
        Record audio and save it to the path
        """

    # @abstractmethod
    # def record_async(self, recording_rate_khz: int, duration: float) -> Optional[queue.Queue]:
    #     """
    #     Record audio and save it to the path
    #     """

    @abstractmethod
    def __enter__(self) -> "Recorder":
        """
        Enter the context manager. Returns self to allow usage with the 'with' statement.
        """
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> Optional[bool]:
        """
        Exit the context manager. This method is meant to handle exception cleanup if necessary.
        """
        pass


class Service(ABC):
    @abstractmethod
    def evaluate_from_microphone(self) -> float:
        """
        Record audio and classify it
        return the probability that the audio contains a baby crying
        """
