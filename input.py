from static_sent_qg import StaticSentQG
from ml_sent_qg.ml_sent_qg import MlSentQG
from ml_para_qg import MlParaQG
from evaluator import Evaluator
from randomizer import Randomizer
from time import sleep
import random

f = open("demo_text.txt")
text = f.read()

static_sent_qg = StaticSentQG(text)
ml_sent_qg = MlSentQG(text)
ml_para_qg = MlParaQG(text)

print "Generating Questions..."

qa_list = static_sent_qg.question_generation() + ml_sent_qg.question_generation() + ml_para_qg.question_generation()

evaluator = Evaluator(qa_list)
evaluator.check_questions()

print "Randomizing Questions..."
sleep(2)

randomizer = Randomizer(qa_list)
qa_list = randomizer.change_order()

for qa in qa_list:
    print "-"*50
    print qa[0]


