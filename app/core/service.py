import hexalog.ports

from app.core import ports


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
    ) -> float:
        """
         Record audio and classify it
        return the probability that the audio contains a baby crying
        """
        self.logger.info("Service beginning to evaluate audio from microphone")
        with self.recorder as recorder:
            audio_file = recorder.record()
        return self.classifier.classify(audio_file)
