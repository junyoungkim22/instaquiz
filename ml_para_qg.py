from textblob import TextBlob

class MlParaQG():
    paragraph = ''
    sent_list = []

    def __init__(self, paragraph):
        self.paragraph = paragraph
        blob = TextBlob(self.paragraph)
        for sent in blob.sentences:
            self.sent_list.append(str(sent))

    def question_generation(self):
        #Fake outputs
        questions = []
        answers = []
        questions.append("What are three fields of study that are related to machine learning?")
        answers.append("computational statistics, mathematical optimization, Data mining")
        return zip(questions, answers)
