import pathlib

import pyaudio
from hexalog.adapters.cli_logger import ColorfulCLILogger
from huggingface_hub import from_pretrained_keras

from app.adapters.classifiers.tensorflow import TensorFlowClassifier
from app.adapters.recorders.pyaudio_recorder import (PyaudioRecorder,
                                                     PyaudioRecordingSettings)
from app.core.service import CryBabyService
from pkg.audio_file_client.adapters.librosa_client import LibrosaClient
from pkg.audio_file_client.core.domain import \
    MelSpectrogramPreprocessingSettings


def main():
    logger = ColorfulCLILogger()
    recorder = PyaudioRecorder(
        logger=logger,
        temp_path=pathlib.Path("/tmp"),
        settings=PyaudioRecordingSettings(
            audio_file_format=pyaudio.paInt16,
            number_of_audio_signals=1,
            frames_per_buffer=1024,
            recording_rate_hz=44100,
            duration_seconds=4,
        ),
    )

    librosa_audio_file_client = LibrosaClient()

    # TODO: The model should keep the mel_spectrogram_preprocessing_settings with it along with the recording settings
    model = from_pretrained_keras("ericcbonet/cry-baby")

    classifier = TensorFlowClassifier(
        model=model,
        audio_file_client=librosa_audio_file_client,
        mel_spectrogram_preprocessing_settings=MelSpectrogramPreprocessingSettings(
            sampling_rate_hz=16000,
            number_of_mel_bands=128,
            duration_seconds=4,
            hop_length=512,
        ),
    )
    # prediction = classifier.classify(pathlib.Path("new_crying.wav"))
    #
    # print(prediction)

    evalualtion = CryBabyService(
        logger=logger, classifier=classifier, recorder=recorder
    ).evaluate_from_microphone()
    # evalualtion
    print(evalualtion)


if __name__ == "__main__":
    main()
