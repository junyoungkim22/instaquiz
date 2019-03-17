from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time
import math
import random
import numpy as np

import squad_loader
from word_index_mapper import WordIndexMapper
from  global_var import PAIRS, DEVICE, MAPPER, TFR, MAX_LENGTH
from txt_token import SOS_TOKEN, EOS_TOKEN
from glove_loader import create_glove_vect_dict, create_emb_layer

class GloveEncoderRNN(nn.Module):
    def __init__(self, hidden_size):
        super(GloveEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = create_emb_layer(hidden_size, True)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.device = DEVICE

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
        
class GloveAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(GloveAttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = MAPPER.n_words
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = create_emb_layer(hidden_size, True)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], hidden[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.device = DEVICE

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], hidden[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, use_attention, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

    loss = 0

    if input_length > max_length:
        input_length = max_length

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_TOKEN]], device=DEVICE)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < TFR else False

    if use_teacher_forcing:
        for di in range(target_length):
            if use_attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            if use_attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_TOKEN:
                break
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(encoder, decoder, n_iters, use_attention, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0    

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = [MAPPER.tensorsFromPair(random.choice(PAIRS)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter-1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, use_attention)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))
            print (print_loss_avg)

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

def evaluate(encoder, decoder, paragraph, use_attention, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = MAPPER.tensorFromParagraph(paragraph)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

        if input_length > max_length:
            input_length = max_length

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
        decoder_input = torch.tensor([[SOS_TOKEN]], device=DEVICE)

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        
        for di in range(max_length):
            if use_attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_TOKEN:
                break
            else:
                decoded_words.append(MAPPER.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, use_attention, n=100):
    for i in range(n):
        pair = random.choice(PAIRS)
        print('T: ', pair[0])
        print('Q: ', pair[1])

        output_words, attentions = evaluate(encoder, decoder, pair[0], use_attention)
        output_question = ' '.join(output_words)
        print('Q? ', output_question)
        print(' ')

def model_train_test(n_iters, print_every):
    hidden_size = 200
    mapper = WordIndexMapper("word_to_index.pkl", "index_to_word.pkl", "word_to_count.pkl")
    #encoder1 = EncoderRNN(mapper.n_words, hidden_size).to(device)
    #decoder1 = DecoderRNN(hidden_size, mapper.n_words).to(device)
    encoder = GloveEncoderRNN(hidden_size).to(DEVICE)
    #attn_decoder1 = AttnDecoderRNN(hidden_size, mapper.n_words).to(device)
    decoder = GloveAttnDecoderRNN(hidden_size).to(DEVICE)
    trainIters(encoder, decoder, n_iters, True, print_every, plot_every=1000)
    evaluateRandomly(encoder, decoder, True)

def load_models(PATH):
    hidden_size = 256
    mapper = WordIndexMapper("word_to_index.pkl", "index_to_word.pkl", "word_to_count.pkl")
    encoder = EncoderRNN(mapper.n_words, hidden_size).to(DEVICE)
    attn_decoder = AttnDecoderRNN(hidden_size, mapper.n_words).to(device)
    encoder.load_state_dict(torch.load(PATH + "50k_encoder"))
    attn_decoder.load_state_dict(torch.load(PATH + "50k_decoder"))
    evaluateRandomly(encoder, attn_decoder, mapper, True)
    
