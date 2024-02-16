import pathlib
import wave
from dataclasses import dataclass

import pyaudio
from hexalog.ports import Logger
from huggingface_hub.file_download import uuid

from app.core import ports


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

    def __enter__(self):
        self.audio_object = pyaudio.PyAudio()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        if self.audio_object is not None:
            self.logger.debug("Terminating audio object")
            self.audio_object.terminate()

    def record(self) -> pathlib.Path:
        file_path = self.temp_path / f"{uuid.uuid4()}.wav"

        audio = pyaudio.PyAudio()

        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024,
        )

        print("recording...")

        frames = []

        for _ in range(0, int(44100 / 1024 * 4)):
            data = stream.read(1024)
            frames.append(data)

        print("finished recording")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        waveFile = wave.open(str(file_path), "wb")
        waveFile.setnchannels(1)
        waveFile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        waveFile.setframerate(44100)
        waveFile.writeframes(b"".join(frames))
        waveFile.close()

        # stream = self._create_stream()
        # frames = self._record(
        #     stream=stream,
        # )
        # stream.stop_stream()
        # stream.close()
        #
        # self._write_to_file(file_path=file_path, frames=frames)
        return file_path

    # def record_async(self, recording_rate_khz: int, duration: float) -> queue.Queue:
    #     """
    #     Record audio and save it to the path
    #     """
    #     audio_recorded_queue = queue.Queue()
    #     stream = self._create_stream()
    #     recording_thread = threading.Thread(
    #         target=self._record_audio,
    #         args=(recording_rate_khz, stream, audio_recorded_queue),
    #     )
    #     recording_thread.start()
    #     self.logger.debug(
    #         "Begin recording audio",
    #         duration=duration,
    #         recording_rate_khz=recording_rate_khz,
    #     )
    #     return audio_recorded_queue

    def _create_stream(self):
        self.logger.debug(
            "Creating audio stream",
            format=self.settings.audio_file_format,
            channels=self.settings.number_of_audio_signals,
            rate=self.settings.recording_rate_hz,
            # frames_per_buffer=self.settings.frames_per_buffer,
        )
        return self.audio_object.open(
            format=self.settings.audio_file_format,
            channels=self.settings.number_of_audio_signals,
            rate=self.settings.recording_rate_hz,
            input=True,
            # frames_per_buffer=self.settings.frames_per_buffer,
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
                self.settings.recording_rate_hz
                / self.settings.frames_per_buffer
                * self.settings.duration_seconds
            ),
        ):
            frames.append(stream.read(self.settings.frames_per_buffer))
        return frames

    def _write_to_file(self, file_path: pathlib.Path, frames: list[bytes]):
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
