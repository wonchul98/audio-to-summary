import os
from pydub import AudioSegment
from pydub.silence import split_on_silence

input_file = "./test_audio/hamburger/hamburger.wav"
output_folder = "./test_audio/hamburger/segments"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

sound_file = AudioSegment.from_wav(input_file)

chunks = split_on_silence(sound_file, min_silence_len=700, silence_thresh=-50)

# 묵음 부분을 제거하고 남은 음성 구간만 연결합니다.
joined_audio = sum(chunks)

segment_index = 1
start = 0
end = 10000
audio_length = len(joined_audio)

while start < audio_length:
    segment = joined_audio[start:end]
    segment_duration = len(segment)

    # 부족한 부분은 0으로 채웁니다.
    if segment_duration < 10000:
        padding = AudioSegment.silent(duration=10000 - segment_duration)
        segment = segment + padding

    segment.export(os.path.join(output_folder, f"segment_{segment_index}.wav"), format="wav")
    segment_index += 1

    start += 10000
    end += 10000
