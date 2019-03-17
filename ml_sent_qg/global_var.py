import squad_loader
import torch

SOS_TOKEN = 0
EOS_TOKEN = 1
ANSS_TOKEN = 2
ANSE_TOKEN = 3

MAX_LENGTH = 100

data = squad_loader.process_file("train-v2.0.json")
PAIRS = squad_loader.prepare_ans_sent_pairs(data)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#TFR is teacher forcing ratio
TFR = 0.5

