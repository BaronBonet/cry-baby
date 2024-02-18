import argparse
import pathlib
import threading

import pyaudio
from hexalog.adapters.cli_logger import ColorfulCLILogger
from huggingface_hub import from_pretrained_keras

from cry_baby.app.adapters.classifiers.tensorflow import TensorFlowClassifier
from cry_baby.app.adapters.recorders.pyaudio_recorder import (
    PyaudioRecorder,
    PyaudioRecordingSettings,
)
from cry_baby.app.core.service import CryBabyService
from cry_baby.pkg.audio_file_client.adapters.librosa_client import LibrosaClient
from cry_baby.pkg.audio_file_client.core.domain import (
    MelSpectrogramPreprocessingSettings,
)

SHUTDOWN_EVENT = threading.Event()


def run_continously(
    logger: ColorfulCLILogger,
    recorder: PyaudioRecorder,
    classifier: TensorFlowClassifier,
):
    service = CryBabyService(logger=logger, classifier=classifier, recorder=recorder)
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
    parser = argparse.ArgumentParser(description="CryBaby CLI Tool")
    parser.add_argument(
        "--run-continuously-tf",
        action="store_true",
        help="Start continuous recording and evaluation",
    )

    parser.add_argument(
        "--run-continuously-tf-lite",
        action="store_true",
        help="Start continuous recording and evaluation with a tensorflow lite model",
    )

    args = parser.parse_args()

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

    librosa_audio_file_client = LibrosaClient()

    if not args.run_continuously_tf:
        logger.info(
            "Please specify a mode to run, using --run-continuously-tf or --run-continuously-tf-lite"
        )

    if args.run_continuously_tf:
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

        run_continously(logger, recorder, classifier)


if __name__ == "__main__":
    main()
