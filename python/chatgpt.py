import openai
from rouge import Rouge
from nltk.translate import meteor_score
import nltk
from blanc import BlancHelp, BlancTune
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def evaluate_summary(summary, reference):
    # Initialize Rouge and Meteor
    rouge = Rouge()

    # Download WordNet if not already downloaded, necessary for meteor
    nltk.download('wordnet', quiet=True) 

    # Tokenize the summary and reference
    tokenized_summary = summary.split()
    tokenized_reference = reference.split()

    # Compute ROUGE scores
    rouge_scores = rouge.get_scores(summary, reference)

    # Compute METEOR score
    meteor_scr = meteor_score.single_meteor_score(tokenized_reference, tokenized_summary)

    return rouge_scores, meteor_scr

openai.api_key = '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'
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "화자별 대화 요약을 생성하는 역할을 모델이다. 화자 별로 요약을 하되 각각 하나씩 요약을 생성하고, 원본 대본 데이터 내에 문맥과 전혀 상관이 없는 내용이 있으면 해당 내용은 무시한다."},
        #data 1
        {"role": "user", "content": """
        원본 대본 데이터)
        해설]트럭 뒤(이른 아침).\n지원, 덕수, 상원 이사를 하는 중이다.\n달랑 보따리 몇 개만 놓여 있다.\n덕수, 무슨 흥에 겨웠는지 기타를 띵땅- 거리며 목 풀고 있다.\n지원과 상원 이불 쓰고 졸고 있다.\n지원](이불 내리고) 아흐.. 아버지. 잠 좀 자요. 잠 좀.\n덕수]아버지라고 하지 말라니까..새 직장 가선 절대 아버지소리 금지다.\n지원]알았어요. 형. 알았으니까 좀 졸기라도 하자구요. 형 말대로 새 직장 첫선 뵈는 날 잠 못자 단체루 눈 뻘개 면접 보면 뭐 좋아요. 상원인 학교두 가야 잖아요..\n덕수](주위를 둘러보다) 차 세워 차!\n지원]아.. 왜 또요?\n덕수](둘러보며) 음... 시심이 불타오르는 풍경 아니냐. 저길 봐라. 아침햇살에 깨어나는 대자연의 숨소리. 뭔가 내 안에서 음표들이 치솟아 오른다.\n지원](졸린 눈으로) 좀 나중에 솟아오르라고 하세요.\n덕수]에라이 자식아! (한대 쥐어박으며) 니가 평생 간판쟁이루 끝난다면, 건.\n가난해서가 아니라 너한테 예술혼이 부족해서야. 임마. 차 세워!\n해설]지원, 기막힌 눈으로 덕수를 보다 운전석과 연결된 창을 두드린다.\n(손잡이 잡고 몸을 앞으로 빼서, 운전석 옆창을 두드리는).\n지원]아저씨! 스톱요! 스톱! 우리 형 변소 급하데요! 스토옵!
        """},
        {"role": "assistant", "content": """
        화자별 요약)
        덕수: 지원에게 아버지라고 말하지 말라고 당부하며, 아침햇살에 영감을 받아 차를 세우라고 한다. 
        지원: 덕수를 아버지대신 형이라고 부른다. 덕수의 노래소리에 잠좀 자자고 소리치지만 덕수에게 한대 얻어맞는다. 차를 세우라는 덕수의 말에 아저씨한테 차를 세우라고 한다. 
        """},
        #data2
        {"role": "user", "content": """
        원본 대본 데이터)
        김경래]그런데 언론 보도를 보면 몇몇 분들의 이름이 거론이 됩니다. 예를 들어서 원희룡 제주지사 그리고 이준석 최고위원도 거론이 되고 진중권 교수 등등이 거론이 되는데 조경태 의원께서는 어떤 분이 됐으면 좋겠다, 어떤 분이 됐으면 좋겠다, 이런 생각 갖고 계시는 분이 있습니까?조경태]앞서 말씀드렸던 대로 이번에 비상대책위가 만약에 꾸려진다면 당을 수습해내는 수습대책위원회의 성격이거든요. 그리고 과거에 비대위가 제가 아까 말씀드렸던 대로 여러 차례 있었다고 하지 않았습니까? 며칠 전에 저한테 그중에 한 분이 오셔서 했던 분 중에 “의원님, 우리 당에는 이미 혁신안이 그 당시 다 나왔다. 그것을 실행만 하면 됩니다.” 이런 이야기를 하더라고요. 지금 이미 당을 혁신하는 비상대책위의 역할은 그동안 20대 국회에서 우리가 축적된 자료들이 있기 때문에 이번에 비상대책위는 수습대책위의 성격을 지니기 때문에 어떤 분이든 맡아서 전당 대회까지 당을 좀 더 수습해내는 그런 역할을 맡으시면 된다, 이런 생각을 합니다. 지금 거론하시는 분들 어떤 분이든 그런 역할을 맡기에는 역량이 충분하다, 이렇게 보고 있습니다.김경래]알겠습니다. 조기 전당 대회 열리면 한 언제쯤으로 예상하세요?조경태]지금 정상적으로 우리가 당헌당규를 보면 8월에 열도록 되어 있거든요. 아마 더불어 민주당 여당도 그즈음에 열리는 것으로 알고 있습니다. 저는 정상적인 전당 대회를 해도 좋을 것 같고요. 그리고 당을 수습한다는 시간적 여유를 준다는 그런 차원에서 그렇습니다. 그렇지 않으면 한두 달 앞당겨서 열어도 좋겠다. 그래서 9월이 정기국회이지 않습니까? 새롭게 출발한다는 의미에서 9월을 넘기지 않도록 하는 것이 좋겠다, 저는 제 입장은 그렇습니다.
        """},
        {"role": "assistant", "content": """
        화자별 요약)
        김경래:  비상대책위원회가 꾸려진다면 누가 역할을 맡기 좋겠냐고 물어보고, 전당대회가 언제 열릴것으로 예상하는지 물어본다. 
        조경태: 과거 비상대책위가 여러 차례가 있었던 바, 비상대책위가 꾸려진다면 수습대책위원회의 성격을 띌 것이라고 하고, 20대 국회에서 쌓인 자료들이 있기 때문에 언론 보도에서 거론된 원희룡, 이준석, 진중권 중 누구든 그 역할을 맡기에 역량이 충분한 것 같으니 9월 정기 국회를 넘기지 않도록 그 전에 정상적인 전당 대회를 열 수 있으면 좋을 것 같다고 답한다. 
        """},
        
        #prompt
        {"role": "user", "content": """
        원본 대본 데이터)
        화자1]2014년 영화 명량의 흥행열기가 대단합니다.\n영화 명량을 보신 많은 분들은 꼭 10년 전에 했던 KBS 드라마 불멸의 이순신을 떠올리시는 분들이 많을 것 같습니다.\n420여 년 전 자신의 모든 것을 바쳐 나라와 백성 그리고 우리 역사를 구했던 인물이순신.\n화자2]살고자 하면 죽을 것이오, 죽고자 하면 살 것이니.\n신에게는 아직 12척의 배가 남아 있습니다.\n화자1]이순신 장군은 생애 그 자체로도 정말 감동적인 삶을 사셨습니다.\n고뇌 그리고 결단 그리고 리더십.\n이런 이순신 장군의 모습들은 오늘날 우리들에게 큰 울림을 던지면서 여전히 우리 시대에 이순신 장군을 기억하게 합니다.\n드라마 불멸의 이순신을 10년 만에 다시 만나는 이유도 여기에 있습니다.\n이순신 장군은 1545년 서울 건천동, 지금의 을지로와 충무로 사이에서 태어나셨고 주로 어린 시절은 외가가 있는 아산에서 그 청소년기를 거치게 됩니다.
        """}
    ],
  temperature=0.3,
  top_p=0.5,
  max_tokens =500
)

answer = response.choices[0].message.content
print(answer)

print("Used tokens:", response['usage']['total_tokens'])


document = "화자1]2014년 영화 명량의 흥행열기가 대단합니다.\n영화 명량을 보신 많은 분들은 꼭 10년 전에 했던 KBS 드라마 불멸의 이순신을 떠올리시는 분들이 많을 것 같습니다.\n420여 년 전 자신의 모든 것을 바쳐 나라와 백성 그리고 우리 역사를 구했던 인물이순신.\n화자2]살고자 하면 죽을 것이오, 죽고자 하면 살 것이니.\n신에게는 아직 12척의 배가 남아 있습니다.\n화자1]이순신 장군은 생애 그 자체로도 정말 감동적인 삶을 사셨습니다.\n고뇌 그리고 결단 그리고 리더십.\n이런 이순신 장군의 모습들은 오늘날 우리들에게 큰 울림을 던지면서 여전히 우리 시대에 이순신 장군을 기억하게 합니다.\n드라마 불멸의 이순신을 10년 만에 다시 만나는 이유도 여기에 있습니다.\n이순신 장군은 1545년 서울 건천동, 지금의 을지로와 충무로 사이에서 태어나셨고 주로 어린 시절은 외가가 있는 아산에서 그 청소년기를 거치게 됩니다."
summary = answer
blanc_help = BlancHelp()
blanc_tune = BlancTune(finetune_mask_evenly=False, show_progress_bar=False)
print(blanc_help.eval_once(document, summary))
print(blanc_tune.eval_once(document, summary))
