from textblob import TextBlob

class MlSentQG():
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
        questions.append("What are mathematial models of sample data in machine learning called?")
        answers.append("training data")
        questions.append("What are three applications of machine learning?")
        answers.append("email filtering, detection of network intruders, computer vision")
        questions.append("Data mining focuses on exploratory data analysis through what method?")
        answers.append("unsupervised learning")

        return (questions, answers)
