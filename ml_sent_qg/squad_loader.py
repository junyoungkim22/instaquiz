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
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace("''", '" ').replace("``", '" ')
                context_tokens = word_tokenize(context)
                print(context)
                print('*'*70)
                print(context_tokens)
                print('-'*80)


process_file("train-v2.0.json")
