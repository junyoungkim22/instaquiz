import re
import random
from textblob import TextBlob

class StaticSentQG:
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
        np = random.choice(sent_blob.noun_phrases)
        pattern = re.compile(np, re.IGNORECASE)
        question = pattern.sub("_______", question)
        return (question, np)

    def question_generation(self):
        question_list = []
        answer_list = []
        for sent in self.sent_list:
            qa_tuple = self.noun_question(sent)
            question_list.append(qa_tuple[0])
            answer_list.append(qa_tuple[1])
        return (question_list, answer_list)
