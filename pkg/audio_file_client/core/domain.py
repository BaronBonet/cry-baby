from dataclasses import dataclass


@dataclass
class MelSpectrogramPreprocessingSettings:
    """
    Class for defining preprocessing settings for the mel spectrogram.

    Attributes:
        sampling_rate_hz: The number of samples of audio carried per second.
                            It defines the resolution at which the audio is
                            digitized. A higher sampling rate provides a
                            higher resolution in the time domain.
                            e.g. 16000

        number_of_mel_bands: The number of Mel frequency bands to use. The
                               Mel scale is a perceptual scale of pitches, so
                               more bands capture finer details in frequency.
                                 e.g. 128

        duration_seconds: The length of the audio signal to process,
                            expressed in seconds. It defines the temporal
                            extent of the audio processing.
                              e.g. 4

        hop_length: The number of samples between the start points of
                      successive frames in time. A smaller hop length
                      increases the overlap between frames and provides finer
                      time resolution.
                        e.g. 512
    """

    sampling_rate_hz: int
    number_of_mel_bands: int
    duration_seconds: int
    hop_length: int

    def __str__(self):
        return (
            f"sr_{self.sampling_rate_hz}__mel_{self.number_of_mel_bands}"
            f"__dur_{self.duration_seconds}__hop_{self.hop_length}"
        )


class LoadError(Exception):
    pass


class UnexpectedDurationError(Exception):
    pass
