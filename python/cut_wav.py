from pydub import AudioSegment
from pydub.silence import split_on_silence

# 오디오 파일을 불러옵니다.
sound_file = AudioSegment.from_wav("hamburger.wav")

# 음성 부분만 분리합니다.
chunks = split_on_silence(sound_file, min_silence_len=700, silence_thresh=-50)

# 10초를 기준으로 자르기 위한 변수입니다.
ten_seconds = 10 * 1000

result = []

# 분리된 오디오 파일을 저장합니다.
for i, chunk in enumerate(chunks):
    # 0.5초보다 짧은 파일은 이전 파일과 합칩니다.
    if len(chunk) < 500:
        if i > 0:
            last_chunk = result[-1]
            result[-1] = last_chunk + chunk
        else:
            result.append(chunk)
    # 10초를 기준으로 파일을 자릅니다.
    else:
        chunk_count = len(chunk) // ten_seconds + 1
        for j in range(chunk_count):
            start = j * ten_seconds
            end = (j + 1) * ten_seconds
            sub_chunk = chunk[start:end]
            sub_chunk.export(f"sep_ham_700ms_50/sep_ham_{i}_{j}.wav", format="wav")