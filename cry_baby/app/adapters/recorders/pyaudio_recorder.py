import math
import pathlib
import queue
import threading
import wave
from dataclasses import dataclass
from typing import Optional

import pyaudio
from hexalog.ports import Logger
from huggingface_hub.file_download import uuid

from cry_baby.app.core import ports


@dataclass
class PyaudioRecordingSettings:
    audio_file_format: int  # e.g. pyaudio.paInt16
    number_of_audio_signals: int  # 1 for mono, 2 for stereo
    frames_per_buffer: int  # e.g. 1024
    recording_rate_hz: int  # e.g. 44100
    duration_seconds: float  # e.g. 4
    """
    frames_per_buffer determines the number of frames (audio samples) processed in each buffer during audio recording.
    It affects the granularity of audio processing and I/O operations.
    A smaller buffer size provides lower latency, but requires more CPU resources and is more prone to buffer overflows.

    With a larger buffersize, the audio system waits to fill the entire buffer before processing it. This results
    in higher latency because there is a delay between the audio input (e.g. recording from a microphone) and the
    and the system response (e.g. playing the audio back through the speakers).

    frames per buffer requires the sample rate to be translated to ms. e.g.
    1024 frames / 16000 frames per second = 0.064 seconds  * 1000 = 64 ms

    recording_rate_khz is the number of samples of audio carried per second.
    It defines the resolution at which the audio is digitized.
    A higher sampling rate provides a higher resolution in the time domain. e.g. 44100

    duration_seconds is the length of the audio signal to process, expressed in seconds.
    """

    def __post_init__(self):
        if self.number_of_audio_signals in [1, 2]:
            return
        raise ValueError("number_of_audio_signals must be 1 or 2")


class PyaudioRecorder(ports.Recorder):
    """
    Using enter and exist for context management this must be used with a with block e.g.
    with PyaudioRecorder(temp_path=temp_path, logger=logger, settings=recorder_settings) as recorder:
        recorder.record(duration=4, recording_rate=44100)
    """

    def __init__(
        self,
        temp_path: pathlib.Path,
        logger: Logger,
        settings: PyaudioRecordingSettings,
    ):
        self.temp_path = temp_path
        self.logger = logger
        self.settings = settings
        self.audio_object = None

    def setup(self):
        if not self.audio_object:
            self.audio_object = pyaudio.PyAudio()
            self.logger.debug("Created pyaudio audio object")

    def tear_down(self):
        if self.audio_object is not None:
            self.logger.debug("Terminating audio object")
            self.audio_object.terminate()
            self.audio_object = None

    def record(self) -> pathlib.Path:
        if not self.audio_object:
            self.setup()

        file_path = self.temp_path / f"{uuid.uuid4()}.wav"

        stream = self._create_audio_stream()
        frames = self._record(
            stream=stream,
        )
        stream.stop_stream()
        stream.close()

        self._write_to_file(file_path=file_path, frames=frames)
        return file_path

    def continuously_record(self) -> Optional[queue.Queue]:
        if not self.audio_object:
            self.setup()

        stream = self._create_audio_stream()
        audio_recorded_queue = queue.Queue()

        recording_thread = threading.Thread(
            target=self._record_continuous,
            args=(stream, audio_recorded_queue),
        )
        self.logger.debug("Starting recording thread")

        recording_thread.daemon = True
        recording_thread.start()
        self.logger.debug("Recording thread started")

        return audio_recorded_queue

    def _create_audio_stream(self) -> pyaudio.Stream:
        if not self.audio_object:
            raise LookupError("Audio object was not created")

        self.logger.debug(
            "Creating audio stream",
            format=self.settings.audio_file_format,
            channels=self.settings.number_of_audio_signals,
            rate=self.settings.recording_rate_hz,
            frames_per_buffer=self.settings.frames_per_buffer,
        )
        return self.audio_object.open(
            format=self.settings.audio_file_format,
            channels=self.settings.number_of_audio_signals,
            rate=self.settings.recording_rate_hz,
            input=True,
            frames_per_buffer=self.settings.frames_per_buffer,
        )

    def _record(self, stream: pyaudio.Stream) -> list[bytes]:
        """
        Record audio and save it to the path
        """
        self.logger.debug(
            "Begin recording audio",
            duration=self.settings.duration_seconds,
            recording_rate_hz=self.settings.recording_rate_hz,
            frames_per_buffer=self.settings.frames_per_buffer,
        )
        frames = []
        for _ in range(
            0,
            int(
                math.ceil(
                    self.settings.recording_rate_hz
                    / self.settings.frames_per_buffer
                    * self.settings.duration_seconds
                )
            ),
        ):
            frames.append(stream.read(self.settings.frames_per_buffer))
        return frames

    def _record_continuous(
        self, stream: pyaudio.Stream, audio_recorded_queue: queue.Queue
    ):
        while True:
            self.logger.debug("Starting to record continuously")
            frames = []
            for _ in range(
                0,
                int(
                    math.ceil(
                        self.settings.recording_rate_hz
                        / self.settings.frames_per_buffer
                        * self.settings.duration_seconds
                    )
                ),
            ):
                frames.append(
                    stream.read(
                        self.settings.frames_per_buffer, exception_on_overflow=False
                    )
                )
            file_path = self.temp_path / f"{uuid.uuid4()}.wav"
            self._write_to_file(file_path, frames)
            audio_recorded_queue.put(file_path)

    def _write_to_file(self, file_path: pathlib.Path, frames: list[bytes]):
        if not self.audio_object:
            raise LookupError("Audio object was not created")
        # record audio in chunks
        waveFile = wave.open(str(file_path), "wb")
        waveFile.setnchannels(self.settings.number_of_audio_signals)
        waveFile.setsampwidth(
            self.audio_object.get_sample_size(self.settings.audio_file_format)
        )
        waveFile.setframerate(self.settings.recording_rate_hz)
        waveFile.writeframes(b"".join(frames))
        waveFile.close()

        self.logger.debug("Written to file", file_path=file_path)
