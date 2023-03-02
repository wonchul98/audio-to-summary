from pytube import YouTube
from moviepy.video.io.VideoFileClip import VideoFileClip

# 유튜브 동영상 다운로드
yt = YouTube('https://www.youtube.com/watch?v=2BN8ZLx8xjY')
video = yt.streams.get_highest_resolution().download()

# 추출할 구간 설정
start_time = 180  # 초단위
end_time = 220  # 초단위

# 동영상 구간 추출
clip = VideoFileClip(video).subclip(start_time, end_time)
clip.write_audiofile('audio.wav')