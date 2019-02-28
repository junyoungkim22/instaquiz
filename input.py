from static_sent_qg import StaticSentQG
from ml_sent_qg import MlSentQG

f = open("demo_text.txt")
text = f.read()

static_sent_qg = StaticSentQG(text)
ml_sent_qg = MlSentQG(text)

for sent in static_sent_qg.question_generation():
    print "-"*50
    print sent
    
for sent in ml_sent_qg.question_generation()[0]:
    print "-"*50
    print sent
