import squad_loader
import torch
import random
from word_index_mapper import WordIndexMapper, make_dictionary

MAX_LENGTH = 30

data = squad_loader.process_file("train-v2.0.json")
PAIRS = squad_loader.prepare_ans_sent_pairs(data)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAPPER = WordIndexMapper("word_to_index.pkl", "index_to_word.pkl", "word_to_count.pkl")

#TFR is teacher forcing ratio
TFR = 0.5

def remake_dictionary():
    reset_maker = WordIndexMapper("word_to_index.pkl", "index_to_word.pkl", "word_to_count.pkl")
    reset_maker.reset()
    make_dictionary()

def para2tensor_test():
    pair = random.choice(PAIRS)
    context, question = pair
    print(MAPPER.normalizeString(question))

    print(pair)
    print(MAPPER.tensorsFromPair(pair))
