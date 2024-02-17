import queue
import threading
from typing import Optional

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
        audio_file = self.recorder.record()
        return self.classifier.classify(audio_file)

    def continously_evaluate_from_microphone(self) -> Optional[queue.Queue]:
        file_written_notification_queue = self.recorder.continuously_record()
        signal_thread = threading.Thread(
            target=self._handle_files_written,
            args=(file_written_notification_queue, self.classifier),
        )
        signal_thread.daemon = True
        signal_thread.start()
        self.logger.info(
            "Service beginning to continuously evaluate audio from microphone"
        )
        self.thread = signal_thread

    def _handle_files_written(
        self, file_written_queue: queue.Queue, classifier: ports.Classifier
    ):
        while True:
            file_path = file_written_queue.get()
            self.logger.debug(f"File written: {file_path}")
            prediction = classifier.classify(file_path)
            self.logger.debug(f"Prediction: {prediction}")

    def stop_continuous_evaluation(self):
        self.recorder.tear_down()
        self.logger.info("Service stopping continuous evaluation")
        print(self.thread)
