import importlib.util
import os
import pathlib
import threading

import pyaudio
from hexalog.adapters.cli_logger import ColorfulCLILogger
from huggingface_hub import from_pretrained_keras, hf_hub_download, login

from cry_baby.app.adapters.recorders.pyaudio_recorder import (
    PyaudioRecorder,
    PyaudioRecordingSettings,
)
from cry_baby.app.adapters.repositories.csv_repo import CSVRepo
from cry_baby.app.core.ports import Repository
from cry_baby.app.core.service import CryBabyService
from cry_baby.pkg.audio_file_client.adapters.librosa_client import LibrosaClient
from cry_baby.pkg.audio_file_client.core.domain import (
    MelSpectrogramPreprocessingSettings,
)

SHUTDOWN_EVENT = threading.Event()


def tensorflow_available():
    tensorflow_spec = importlib.util.find_spec("tensorflow")
    return tensorflow_spec is not None


def tflite_runtime_available():
    tflite_spec = importlib.util.find_spec("tflite_runtime")
    return tflite_spec is not None


def run_continously(
    logger: ColorfulCLILogger,
    recorder: PyaudioRecorder,
    classifier,
    repository: Repository,
):
    service = CryBabyService(
        logger=logger, classifier=classifier, recorder=recorder, repository=repository
    )
    logger.info("Starting to continously evaluate from microphone")
    while not SHUTDOWN_EVENT.is_set():
        try:
            service.continously_evaluate_from_microphone()
            logger.info("Press ctr+c to stop")
            SHUTDOWN_EVENT.wait()
        except KeyboardInterrupt:
            logger.info("Stopping CryBabyService")
            service.stop_continuous_evaluation()
            logger.debug("CryBabyService has stopped")
            SHUTDOWN_EVENT.set()
            logger.debug("Exiting")


def main():
    logger = ColorfulCLILogger()
    temp_path = pathlib.Path("/tmp")
    settings = PyaudioRecordingSettings(
        audio_file_format=pyaudio.paInt16,
        number_of_audio_signals=1,
        frames_per_buffer=1024,
        recording_rate_hz=44100,
        duration_seconds=4,
    )
    recorder = PyaudioRecorder(logger=logger, temp_path=temp_path, settings=settings)
    repository = CSVRepo(csv_file_path=pathlib.Path("predictions.csv"))

    librosa_audio_file_client = LibrosaClient()

    mel_spectrogram_preprocessing_settings = MelSpectrogramPreprocessingSettings(
        sampling_rate_hz=16000,
        number_of_mel_bands=128,
        duration_seconds=4,
        hop_length=512,
    )

    # TODO: it's horrible to import the classifiers here
    if tensorflow_available():
        from cry_baby.app.adapters.classifiers.tensorflow import TensorFlowClassifier

        model = from_pretrained_keras("ericcbonet/cry-baby")
        classifier = TensorFlowClassifier(
            model=model,
            audio_file_client=librosa_audio_file_client,
            mel_spectrogram_preprocessing_settings=mel_spectrogram_preprocessing_settings,
        )
        logger.info("Using TensorFlow classifier.")
    elif tflite_runtime_available():
        from cry_baby.app.adapters.classifiers.tf_lite import TFLiteClassifier

        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            logger.error("HUGGINGFACE_TOKEN does not exist in the environment")
            return
        login(token=token)
        model_path = hf_hub_download(
            repo_id="ericcbonet/cry_baby_lite", filename="model.tflite"
        )
        classifier = TFLiteClassifier(
            mel_spectrogram_preprocessing_settings,
            librosa_audio_file_client,
            pathlib.Path(model_path),
        )
        logger.info("Using TensorFlow Lite classifier.")
    else:
        logger.error("No compatible TensorFlow or TensorFlow Lite installation found.")
        return

    run_continously(logger, recorder, classifier, repository)


if __name__ == "__main__":
    main()
