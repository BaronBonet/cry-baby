import pathlib
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
