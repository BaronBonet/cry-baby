import pathlib
import queue
from abc import ABC, abstractmethod
from typing import Optional

from cry_baby.pkg.audio_file_client.core.domain import (
    MelSpectrogramPreprocessingSettings,
)


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

    @abstractmethod
    def continuously_record(self) -> Optional[queue.Queue]:
        """
        Continuously record audio and save it to the path
        returns a queue of the audio recorded and an event to stop the recording

        Usage:
        ```
        with recorder:
            audio_queue = recorder.continuously_record()
        ```
        """

    @abstractmethod
    def setup(self):
        """
        Setup the required resources for recording
        """

    @abstractmethod
    def tear_down(self):
        """
        Tear down the required resources for recording
        """


class Service(ABC):
    @abstractmethod
    def evaluate_from_microphone(self) -> float:
        """
        Record audio and classify it
        return the probability that the audio contains a baby crying
        """

    @abstractmethod
    def continously_evaluate_from_microphone(self) -> Optional[queue.Queue]:
        """
        Continuously record audio and classify it
        return a queue of the probabilities that the audio contains a baby crying
        """

    @abstractmethod
    def stop_continuous_evaluation(self):
        """
        Stop the continuous evaluation
        """
