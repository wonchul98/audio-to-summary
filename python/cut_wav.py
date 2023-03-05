import os
import numpy as np
import librosa
import math


class VoiceActivityDetector:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.samples, self.sample_rate = librosa.load(audio_file, sr=None, mono=True)

    def get_voice_regions(self, window_duration=0.5, power_threshold=0.5, plot=False):
        samples_per_window = int(self.sample_rate * window_duration)
        overlap = 0.5

        # Calculate the power spectrum of the audio signal
        power_spectrum = np.square(self.samples)

        # Apply a moving average filter to smooth the power spectrum
        filter_size = int(self.sample_rate * 0.04)
        filter_b = np.ones(filter_size) / filter_size
        power_spectrum = np.convolve(power_spectrum, filter_b, mode='same')

        # Apply a median filter to remove outliers
        filter_size = int(self.sample_rate * 0.4)
        power_threshold = power_threshold * np.median(power_spectrum)
        power_spectrum = np.minimum(power_spectrum, np.median(power_spectrum) * 2)
        power_spectrum = np.convolve(power_spectrum, np.ones(filter_size), mode='same')

        # Find the start and end indices of the regions with high power
        is_voice_region = power_spectrum > power_threshold
        is_voice_region[:int(self.sample_rate * 0.01)] = False
        is_voice_region[-int(self.sample_rate * 0.01):] = False

        voice_samples = np.where(is_voice_region)[0]
        diff = np.diff(voice_samples)
        diff = np.insert(diff, 0, 0)
        splits = np.where(diff > samples_per_window)[0]

        regions = []
        for i in range(len(splits)):
            if i == 0:
                start = voice_samples[0]
            else:
                start = voice_samples[splits[i - 1] + 1]

            end = voice_samples[splits[i]] + 1
            regions.append([start, end])

        # Plot the power spectrum and voice regions (optional)
        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(15, 5))
            plt.plot(np.arange(len(power_spectrum)) / self.sample_rate, power_spectrum)
            ymin, ymax = plt.ylim()
            for region in regions:
                plt.fill_between(np.arange(region[0], region[1]) / self.sample_rate, ymin, ymax, alpha=0.2, color='red')
            plt.ylim(ymin, ymax)
            plt.xlabel('Time (s)')
            plt.ylabel('Power')
            plt.title('Power Spectrum')
            plt.show()

        return regions

    def save_voice_regions(y, sr, regions, file_prefix, min_length=0.5, max_length=10):
        '''
    분리된 음성 파일을 저장하는 함수

    Args:
    - y: 분리된 음성 데이터
    - sr: sampling rate
    - regions: 분리된 음성 구간 정보 (start, end)
    - file_prefix: 저장할 파일 이름의 접두어
    - min_length: 최소 길이 (초)
    - max_length: 최대 길이 (초)
    '''

    # 최소 길이 이하로 분리된 음성 구간은 이전 파일과 합치기 위한 변수
        prev_region = None
        prev_audio = None

        for i, region in enumerate(regions):
            # 구간의 길이 계산
            region_length = region[1] - region[0]

            # 최소 길이 이하인 경우 이전 파일과 합침
            if region_length < min_length and prev_region is not None:
                prev_audio = librosa.util.fix_length(prev_audio, prev_audio.shape[0]+y[region[0]:region[1]].shape[0])
                prev_audio[prev_region[1]:prev_region[1]+region_length] += y[region[0]:region[1]]
                prev_region = (prev_region[0], prev_region[1]+region_length)

            # 최대 길이 이상인 경우 10초 간격으로 나누어 저장
            elif region_length > max_length:
                num_subregions = region_length // max_length
                for j in range(num_subregions):
                    subregion_start = region[0] + j*max_length*sr
                    subregion_end = subregion_start + max_length*sr
                    subregion = (subregion_start, subregion_end)
                    subregion_length = subregion_end - subregion_start
                    subregion_audio = y[subregion_start:subregion_end]

                    # 저장할 파일 이름 생성
                    file_name = f"{file_prefix}_{i}_{j}.wav"
                    file_path = os.path.join("sep_ham", file_name)

                    # librosa.output.write_wav 함수를 사용하여 파일 저장
                    librosa.output.write_wav(file_path, subregion_audio, sr=sr, norm=False)

            # 일반적인 경우 파일 저장
            else:
                region_audio = y[region[0]:region[1]]

                # 저장할 파일 이름 생성
                file_name = f"{file_prefix}_{i}.wav"
                file_path = os.path.join("sep_ham", file_name)

                # librosa.output.write_wav 함수를 사용하여 파일 저장
                librosa.output.write_wav(file_path, region_audio, sr=sr, norm=False)

                prev_region = region
                prev_audio = region_audio

vad = VoiceActivityDetector("hamburger.wav")
vad.get_voice_regions()
vad.save_voice_regions("sep_ham")