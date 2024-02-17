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
        with self.recorder as recorder:
            audio_file = recorder.record()
        return self.classifier.classify(audio_file)

    def continously_evaluate_from_microphone(self) -> Optional[queue.Queue]:
        with self.recorder as recorder:
            file_written_notification_queue = recorder.continuously_record()
            signal_thread = threading.Thread(
                target=_handle_files_written,
                args=(file_written_notification_queue, self.classifier),
            )
            signal_thread.start()
            self.logger.info("Service beginning to continuously evaluate audio from microphone")


def _handle_files_written(
    file_written_queue: queue.Queue, classifier: ports.Classifier
):
    while True:
        file_path = file_written_queue.get()
        print(f"File written: {file_path}")
        prediction = classifier.classify(file_path)
        print(f"Prediction: {prediction}")
