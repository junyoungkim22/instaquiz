from static_sent_qg import StaticSentQG
from ml_sent_qg import MlSentQG
from ml_para_qg import MlParaQG

f = open("demo_text.txt")
text = f.read()

static_sent_qg = StaticSentQG(text)
ml_sent_qg = MlSentQG(text)
ml_para_qg = MlParaQG(text)

qa_list = static_sent_qg.question_generation() + ml_sent_qg.question_generation() + ml_para_qg.question_generation()

for qa in qa_list:
    print "-"*50
    print qa[0]
