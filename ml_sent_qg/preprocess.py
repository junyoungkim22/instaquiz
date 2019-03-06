import torch 

SOS_TOKEN = 0
EOS_TOKEN = 1
ANSS_TOKEN = 2
ANSE_TOKEN = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def indexesFromParagraph(indexer, paragraph):
    ret = []
    for sent in paragraph.split('.'):
        ret += indexesFromSentence(indexer, sent)
    return ret

def indexesFromSentence(indexer, sentence):
    ret = []
    for word in sentence.split(' '):
        try:
            ret.append(indexer.word2index[word])
        except KeyError:
            continue
    return ret

def tensorFromParagraph(indexer, paragraph):
    indexes = indexesFromParagraph(indexer, paragraph)
    indexes.append(EOS_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorFromSentence(indexer, sentence):
    indexes = indexesFromSentence(indexer, sentence)
    indexes.append(EOS_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)
