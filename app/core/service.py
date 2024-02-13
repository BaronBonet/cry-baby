import hexalog.ports

from app.core import ports
from pkg.audio_file_client.core.domain import \
    MelSpectrogramPreprocessingSettings


class CryBabyService(ports.Service):
    def __init__(
        self,
        logger: hexalog.ports.Logger,
        classifier: ports.Classifier,
        recorder: ports.Recorder,
    ):
        self.logger = logger
        self.classifier = classifier
        self.recorder = recorder

    def evaluate_from_microphone(
        self,
        recording_rate_khz: int,
        mel_spectrogram_preprocessing_settings: MelSpectrogramPreprocessingSettings,
    ) -> float:
        """
         Record audio and classify it
        return the probability that the audio contains a baby crying
        """
        self.logger.info("Service beginning to evaluate audio from microphone")
        with self.recorder as recorder:
            audio_file = recorder.record(
                recording_rate_khz,
                mel_spectrogram_preprocessing_settings.duration_seconds,
            )
        return self.classifier.classify(
            audio_file, mel_spectrogram_preprocessing_settings
        )
