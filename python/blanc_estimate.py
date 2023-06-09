
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
nltk.download('punkt')
sums = glob.glob('C:/Users/Administrator/Desktop/졸업 프로젝트/audio-to-summary/python/test_audio/multi_speaker/대본 요약/*.txt')
docs = glob.glob('C:/Users/Administrator/Desktop/졸업 프로젝트/audio-to-summary/python/test_audio/multi_speaker/대본/*.txt')
documents = []
summaries = []
all_blanc_scores = []

for i in range(0,10):
    print(i)
    doc_path = docs[i]
    sum_path = sums[i]
    
    with open(doc_path, 'r', encoding='utf-8') as doc:
        doc_text = doc.read()
    with open(sum_path, 'r', encoding='utf-8') as summ:
        sum_text = summ.read()
    #print(doc_text)
    #print(sum_text)
    documents.append(doc_text)
    summaries.append(sum_text)
    
    
# Calculate the total average
blanc_help = BlancHelp()
#print(f"documents: ", documents)
#print(f"summaries: ", summaries)
#print("documents: ")
#for document in documents:
#    print(document)
#print("summaries: ")
#for summary in summaries:
#    print(summary)

blanc_scores = blanc_help.eval_pairs(documents, summaries)
print(blanc_scores)
total_average_blanc_score = sum(blanc_scores) / len(blanc_scores)
print(total_average_blanc_score)
