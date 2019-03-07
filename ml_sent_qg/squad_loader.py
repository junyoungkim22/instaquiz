import simplejson as json
from tqdm import tqdm
import spacy
import sys

'''
This file is taken and modified from
https://github.com/deepakkumar1984/QANet2/blob/master/prepro.py
'''

nlp = spacy.blank("en")

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]

def process_file(file_name):
    with open("data/" + file_name, "r") as data_file:
        source = json.load(data_file)
        ret = []
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace("''", '" ').replace("``", '" ')
                #context_tokens = word_tokenize(context)
                context_qas = []
                for qa in para["qas"]:
                    question = qa["question"].replace("''", '" ').replace("``", '" ')
                    ans_txt_pos = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        ans_txt_pos.append((answer_text, (answer_start, answer_end)))
                    context_qas.append((question, ans_txt_pos))
                ret.append((context, context_qas))
        return ret

def prepare_pairs():
    data = process_file("train-v2.0.json")
    pairs = []
    for context, context_qas in data:
        for question, answers in context_qas:
            pairs.append((context, question))
    return pairs


def test():
    data = process_file("train-v2.0.json")
    i = 0
    for context, context_qas in data:
        print(context)
        print('*'*80)
        for question, answers in context_qas:
            print(question)
            print('-'*80)
            for txt, (start, end) in answers:
                print(txt)
                print(start)
                print(end)
                print('&'*80)
        print("\n")
        i += 1
        if(i == 20):
            break
