import pathlib

import numpy as np
import tflite_runtime.interpreter as tflite

from cry_baby.app.core import ports
from cry_baby.pkg.audio_file_client.core.domain import (
    MelSpectrogramPreprocessingSettings,
)
from cry_baby.pkg.audio_file_client.core.ports import AudioFileClient


class TFLiteClassifier(ports.Classifier):
    """
    Classify the audio file with the TFLite model
    This will not work on macOS
    """

    def __init__(
        self,
        mel_spectrogram_preprocessing_settings: MelSpectrogramPreprocessingSettings,
        audio_file_client: AudioFileClient,
        model_path: pathlib.Path,
    ):
        self.model_path = model_path
        self.audio_file_client = audio_file_client
        self.mel_spectrogram_preprocessing_settings = (
            mel_spectrogram_preprocessing_settings
        )

    def classify(self, path_to_audio_file: pathlib.Path) -> float:
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
