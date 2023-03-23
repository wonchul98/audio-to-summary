from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
model = separator.from_hparams(source="speechbrain/sepformer-libri2mix", savedir='pretrained_models/sepformer-libri2mix',run_opts={"device":"cuda"})

import os

# Get the current directory path
dir_path = os.getcwd()

# Get the list of files in the current directory
files = os.listdir(dir_path)

# Print the list of files
print("Files in the current directory:")
for file in files:
    print(file)
    
import os
import re

# 폴더 경로
folder_path = "./cut_wav"

# 파일명 패턴
pattern = r"sep_ham_(\d+)_(\d+)\.wav"

# 파일명에서 i, j 추출하는 함수
def extract_ij(filename):
    match = re.match(pattern, filename)
    if match:
        i, j = match.groups()
        return int(i), int(j)
    else:
        return tuple()

# 폴더 내 wav 파일 찾기
wav_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".wav")]

# 파일명에서 i, j 추출하여 정렬
wav_files.sort(key=extract_ij)

new_folder_path = os.path.join(folder_path, "2sep_output")
# 결과 출력
for filename in wav_files:
    path = os.path.join(folder_path, filename)
    est_sources = model.separate_file(path=path)
    
    for k in range(2):
        out_filename = f"{filename.split('.')[0]}_{k}.wav"
        out_path = os.path.join(new_folder_path, out_filename)
        torchaudio.save(out_path, est_sources[:, :, k].detach().cpu(), 8000)