from textblob import TextBlob
from abc import ABC, abstractmethod

class MlSentQG(ABC):
    paragraph = ''
    sent_list = []

    def __init__(self, paragraph):
        self.paragraph = paragraph
        blob = TextBlob(self.paragraph)
        for sent in blob.sentences:
            self.sent_list.append(str(sent))

    @abstractmethod
    def question_generation(self):
        pass
