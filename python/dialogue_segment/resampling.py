import os
import numpy as np
import soundfile as sf
import librosa

def resample_audio_files(input_folder, output_folder, target_sr=16000):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".wav"):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_folder, file)

                y, sr = librosa.load(input_file_path, sr=44100)
                resampled_audio = librosa.resample(y, sr, target_sr)

                sf.write(output_file_path, resampled_audio, target_sr, subtype='PCM_16')

input_folder = "separated_segments"
output_folder = "resampled_segments"
resample_audio_files(input_folder, output_folder)
