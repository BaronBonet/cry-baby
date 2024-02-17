import pathlib

import numpy as np
import tflite

from app.core import ports
from pkg.audio_file_client.core.domain import (
    MelSpectrogramPreprocessingSettings, UnexpectedDurationError)
from pkg.audio_file_client.core.ports import AudioFileClient


class TFLiteClassifierBabyCrying(ports.Classifier):
    """
    Classify the audio file with the TFLite model
    This will not work on macOS
    """

    def __init__(
        self,
        model_path: pathlib.Path,
        audio_file_client: AudioFileClient,
        mel_spectrogram_preprocessing_settings: MelSpectrogramPreprocessingSettings,
    ):
        self.model_path = model_path
        self.audio_file_client = audio_file_client
        self.mel_spectrogram_preprocessing_settings = (
            mel_spectrogram_preprocessing_settings
        )

    def classify(self, path_to_audio_file: pathlib.Path) -> float:
        if not path_to_audio_file.exists() or not path_to_audio_file.is_file():
            raise FileNotFoundError

        # Check that the duration of the audio file is as long as the pre_processing_settings.duration_seconds
        if (
            duration := self.audio_file_client.get_duration(
                path_to_audio_file,
                hop_length=self.mel_spectrogram_preprocessing_settings.hop_length,
                sampling_rate_hz=self.mel_spectrogram_preprocessing_settings.sampling_rate_hz,
            )
            != self.mel_spectrogram_preprocessing_settings.duration_seconds
        ):
            raise UnexpectedDurationError(
                f"Audio file {path_to_audio_file} has duration {duration} seconds, "
                f"but the pre_processing_settings.duration_seconds is"
                f" {self.mel_spectrogram_preprocessing_settings.duration_seconds}"
            )

        # Extract features
        mel_spec = self.audio_file_client.extract_mel_spectrogram(
            path_to_audio_file, self.mel_spectrogram_preprocessing_settings
        )

        interpreter = tflite.Interpreter(model_path=str(self.model_path))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        mel_spec = np.array(mel_spec, dtype=np.float32)
        mel_spec = np.expand_dims(mel_spec, axis=0)  # Add a batch dimension
        mel_spec = np.expand_dims(mel_spec, axis=-1)  # Add a channel dimension

        interpreter.set_tensor(input_details["index"], mel_spec)

        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details["index"])

        if prediction.shape != (1, 1):
            raise ValueError(
                f"Expected prediction shape to be (1, 1), but got {prediction.shape}"
            )

        return prediction[0][0]
