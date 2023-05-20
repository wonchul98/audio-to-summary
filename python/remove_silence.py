import collections
import contextlib
import sys
import wave
import webrtcvad
import numpy as np
from scipy.io.wavfile import write
import numpy as np
import webrtcvad
import wave
import contextlib

import numpy as np
import webrtcvad
import wave
import contextlib

def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

def remove_silence(path):
    audio, sample_rate = read_wave(path)
    raw_samples = np.frombuffer(audio, dtype=np.int16)

    vad = webrtcvad.Vad(3) # Set aggressiveness from 0 to 3
    frame_duration = 30  # ms
    frame_size = int(sample_rate * (frame_duration / 1000.0))
    frames = []

    # Make sure that the length of the audio is a multiple of the frame size
    raw_samples = raw_samples[:len(raw_samples) // frame_size * frame_size]

    for start in np.arange(0, len(raw_samples), frame_size):
        stop = min(start + frame_size, len(raw_samples))
        is_speech = vad.is_speech(raw_samples[start:stop].tobytes(), sample_rate=sample_rate)

        if is_speech:
            frames.append(raw_samples[start:stop])

    return np.concatenate(frames), sample_rate



def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)




# Use the function
audio, sample_rate = remove_silence('sample_for_vad1_PCM.wav')
write_wave('vad_no_silence.wav', audio, sample_rate)