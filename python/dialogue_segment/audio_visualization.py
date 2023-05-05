import librosa
import librosa.display
import matplotlib.pyplot as plt

# 오디오 파일 로드
audio_file = 'dialogue.wav'
y, sr = librosa.load(audio_file, sr=None)

# wavplot으로 오디오 시각화
plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Waveplot - hamburger.wav')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.show()