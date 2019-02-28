from static_sent_qg import StaticSentQG
from ml_sent_qg import MlSentQG
from ml_para_qg import MlParaQG

f = open("demo_text.txt")
text = f.read()

static_sent_qg = StaticSentQG(text)
ml_sent_qg = MlSentQG(text)
ml_para_qg = MlParaQG(text)

for sent in static_sent_qg.question_generation():
    print "-"*50
    print sent
    
for sent in ml_sent_qg.question_generation()[0]:
    print "-"*50
    print sent

for sent in ml_para_qg.question_generation()[0]:
    print "-"*50
    print sent
