import nltk
import re
from textblob import TextBlob

class Sentence_QG:
    paragraph = ''
    sent_list = []
    def __init__(self, paragraph):
        self.paragraph = paragraph
        blob = TextBlob(self.paragraph)
        for sent in blob.sentences:
            self.sent_list.append(str(sent))
            
    def noun_question(self, sentence):
        sent_blob = TextBlob(sentence)
        noun_list = sent_blob.noun_phrases
        question = sentence
        for np in sent_blob.noun_phrases:
            pattern = re.compile(np, re.IGNORECASE)
            question = pattern.sub("_______", question)
        return question

    def question_generation(self):
        question_list = []
        for sent in self.sent_list:
            question_list.append(self.noun_question(sent))
        return question_list

f = open("demo_text.txt")
text = f.read()

sent_qg = Sentence_QG(text)

print sent_qg.question_generation()

blob = TextBlob(text)
