from static_sent_qg import StaticSentQG

f = open("demo_text.txt")
text = f.read()

sent_qg = StaticSentQG(text)

for sent in sent_qg.question_generation():
    print "-"*50
    print sent
