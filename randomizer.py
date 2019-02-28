import random

class Randomizer:
    qa_list = []

    def __init__(self, qa_list):
        self.qa_list = qa_list

    def change_order(self):
        return random.sample(self.qa_list, len(self.qa_list))
