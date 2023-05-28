import openai
from rouge import Rouge
from nltk.translate import meteor_score
import nltk
from blanc import BlancHelp, BlancTune
import warnings
import json
import re
from collections import defaultdict
import os
import glob
warnings.filterwarnings("ignore", category=FutureWarning)
openai.api_key = '<key>'
filepaths = glob.glob('D:/023.방송 콘텐츠 대본 요약 데이터/01.데이터/1.Training/라벨링데이터/TL1/culture/20per/*.json')

filepaths = filepaths[91:100]  #
all_blanc_scores = []
i = 1
for filepath in filepaths:
    print(filepath)
    # json 파일 열기
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        passage = data['Meta']['passage']
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """
                Your task is to summarize each speaker's utterances from the given text. Please ensure that:
                1. A summary is generated for each speaker.
                2. Each speaker's summary should be based on all of their utterances, producing only a single summary for each speaker.
                3. Ignore any sentences that are not associated with any context.
                4. Since post-processing is required with the '\{speaker\}:\{content summary\}' format, it must be followed. 
                5. Do not put special symbols such as '-' in front of 'speaker:'.
                6. '\{speaker\} is \{summary of speech\}It is not a form like , but it must be in the form of '\{speaker\}:\{speech summary\}'
                """},
                #data
                {"role": "user", "content": """
                원본 대본 데이터)
                인영]편의점 가자\n태준]그래\n인영]먹고 싶은거 있어?\n태준]라면 먹고싶어\n인영]내가 사줄게]\n태준]아니야 괜찮아\n
                """},
                {"role": "assistant", "content": """
            인영: 화자2에게 편의점에 가자고 제안하며, 라면을 사주겠다고 한다. 
            태준: 화자1과 편의점에 가기로 하며 화자1이 라면을 사준다는 제안을 거절한다.
                """},
                
                #prompt
                {"role": "user", "content": "원본 대본 데이터)\n" + passage }
            ],
        temperature=0.3,
        top_p=0.5,
        max_tokens =500
        )


        answer = response.choices[0].message.content
        
        #print(f"answer: \n",{answer})
        #print("Used tokens:", response['usage']['total_tokens'])

        speakers = {}
        current_speaker = None
        delimiters = ['\n', '.', '?', '!']

        # 문자열을 줄바꿈으로 분할
        lines = re.split('|'.join(map(re.escape, delimiters)), passage)

        for line in lines:
            # 각 줄의 시작 부분에서 화자를 찾음
            speaker_match = re.search(r'(.*?)\]', line)
            if speaker_match:
                # 화자를 추출하고 발언을 분리
                current_speaker = speaker_match.group(1)
                speech = line[speaker_match.end():].strip()
            else:
                # 화자가 명시되지 않은 줄은 이전 화자의 발언으로 간주
                speech = line.strip()
                
            if current_speaker:
                # 발언 내용을 추가하거나 새로 생성
                if current_speaker in speakers:
                    speakers[current_speaker] += [speech]
                else:
                    speakers[current_speaker] = [speech]
        #print(f"speakers before preprocessing: \n", speakers)
        speakers = {speaker: [speech + '.' if not speech.endswith('.') else speech for speech in speeches if speech] for speaker, speeches in speakers.items()}
        speakers = {speaker: ' '.join(speeches).strip() for speaker, speeches in speakers.items()}
        #print(f"speakers: \n", speakers)
        
        sum_speakers = {}
        sum_current_speaker = None

        lines = re.split(r'\n', answer.strip())  # remove leading/trailing white spaces and split by new line

        for line in lines:
            speaker_match = re.search(r'^(.*?):', line)  # find speaker at the start of the line
            if speaker_match:
                sum_current_speaker = speaker_match.group(1).strip()  # extract the speaker
                sum_speech = line[speaker_match.end():].strip()  # separate the speech
            else:
                sum_speech = line.strip()

            if sum_current_speaker:
                if sum_current_speaker in sum_speakers:
                    sum_speakers[sum_current_speaker] += " " + sum_speech
                else:
                    sum_speakers[sum_current_speaker] = sum_speech

        

        common_speakers = set(speakers.keys()) & set(sum_speakers.keys())

        filtered_speakers = {speaker: speakers[speaker] for speaker in common_speakers}
        filtered_sum_speakers = {speaker: sum_speakers[speaker] for speaker in common_speakers}

        # 발언자의 이름 순서로 정렬
        sorted_speakers = dict(sorted(filtered_speakers.items(), key=lambda item: item[0]))
        sorted_sum_speakers = dict(sorted(filtered_sum_speakers.items(), key=lambda item: item[0]))
        #print(f"common_speaker: ", common_speakers)

        documents = list(sorted_speakers.values())
        summaries = list(sorted_sum_speakers.values())

        #documents = [' '.join(speech) for speech in sorted_speakers.values()]

        # 결과를 확인합니다.
        #print(f"화자 별 발화 내용:\n", documents)
        #print(f"화자 별 요약 내용:\n", summaries)
        blanc_help = BlancHelp()

        blanc_scores = blanc_help.eval_pairs(documents, summaries)
        print(blanc_scores)
        all_blanc_scores.extend(blanc_scores)
        print(f"i: ", i, " sum: ", sum(all_blanc_scores), " len: ", len(all_blanc_scores))
        i += 1
    
# Calculate the total average
total_average_blanc_score = sum(all_blanc_scores) / len(all_blanc_scores)
print("Total Average Blanc Score: ", total_average_blanc_score)
