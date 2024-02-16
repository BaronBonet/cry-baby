import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from app.core import ports
from pkg.audio_file_client.core.domain import \
    MelSpectrogramPreprocessingSettings
from pkg.audio_file_client.core.ports import AudioFileClient


class TensorFlowClassifier(ports.Classifier):
    def __init__(
        self,
        mel_spectrogram_preprocessing_settings: MelSpectrogramPreprocessingSettings,
        model: Model,
        audio_file_client: AudioFileClient,
    ):
        self.model = model
        self.audio_file_client = audio_file_client
        self.mel_spectrogram_preprocessing_settings = mel_spectrogram_preprocessing_settings

    def classify(
        self,
        path_to_audio_file: pathlib.Path,
    ) -> float:
        # Extract features
        mel_spec = self.audio_file_client.extract_mel_spectrogram(
            path_to_audio_file, self.mel_spectrogram_preprocessing_settings
        )

        mel_spec = np.expand_dims(mel_spec, axis=-1)  # Add a channel dimension

        mel_spec = np.expand_dims(mel_spec, axis=0)  # Add a batch dimension

        # Predict
        prediction: np.ndarray = self.model.predict(mel_spec)

        if prediction.shape != (1, 1):
            raise ValueError(
                f"Expected prediction shape to be (1, 1), but got {prediction.shape}"
            )

        return prediction[0][0]
