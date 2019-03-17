import squad_loader
import torch
from word_index_mapper import WordIndexMapper

MAX_LENGTH = 100

data = squad_loader.process_file("train-v2.0.json")
PAIRS = squad_loader.prepare_ans_sent_pairs(data)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAPPER = WordIndexMapper("word_to_index.pkl", "index_to_word.pkl", "word_to_count.pkl")

#TFR is teacher forcing ratio
TFR = 0.5
