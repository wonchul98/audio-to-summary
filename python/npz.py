import numpy as np


# npz 파일 로드
data = np.load('toy_training_data.npz')
# npz 파일 내 변수 확인
print(data.files)

import scipy.io.wavfile as wav

# WAV 파일 생성을 위한 NumPy 배열 추출
signal = data['train_sequence']
print(signal.shape)
scaled = np.int16(signal / np.max(np.abs(signal)) * (2**15)) #정규화 되어있는 파일 스케일 업
scaled = np.ravel(scaled, order = 'C') #2차원 scaled배열 1차원으로 
print(scaled.shape)
print(scaled)
id = data['train_cluster_id']
print(id.shape)
#print(type(id[0][0]))
#print(id[:][0])
    

#print(id[:500])
#print(np.unique(id, return_index = True))
# NumPy 배열을 WAV 파일로 저장
wav.write('output.wav', 16000, scaled) #.wav파일로 변환 2번째 인자가 sampling rate

