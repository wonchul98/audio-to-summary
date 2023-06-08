import os
from pydub import AudioSegment
from pydub.silence import split_on_silence

def save_segment(segment, index, folder):
    segment.export(os.path.join(folder, f"segment_{index}.wav"), format="wav")
    
for n in range(3,11):
    input_file = "./test_audio/single_speaker/" + str(n) +"/single.wav"
    output_folder = "./test_audio/single_speaker/" + str(n) +"/segments"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 오디오 파일을 불러옵니다.
    sound_file = AudioSegment.from_wav(input_file)

    # 음성 부분만 분리합니다.
    chunks = split_on_silence(sound_file, min_silence_len=700, silence_thresh=-50)

    # segments 폴더를 생성합니다.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 길이가 0.5초보다 작은 경우 이전 파일과 합치고, 20초가 넘으면 자르는 코드
    merged_chunks = []
    for chunk in chunks:
        if len(chunk) < 500:  # 0.5초보다 작으면
            if merged_chunks:
                merged_chunks[-1] += chunk  # 이전 파일과 합치기
        elif len(chunk) > 10000:  # 20초가 넘으면
            while len(chunk) > 10000:
                part, chunk = chunk[:10000], chunk[10000:]
                merged_chunks.append(part)
        else:
            merged_chunks.append(chunk)

    # segments 폴더에 segment_{i}.wav 형식으로 분리된 오디오를 저장합니다.
    for i, chunk in enumerate(merged_chunks):
        save_segment(chunk, i, output_folder)