from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time
import math
import random
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

import squad_loader
from word_index_mapper import WordIndexMapper
from  global_var import PAIRS, DEVICE, MAPPER, TFR, MAX_LENGTH
from txt_token import SOS_TOKEN, EOS_TOKEN
from glove_loader import create_glove_vect_dict, create_emb_layer

class GloveEncoderRNN(nn.Module):
    def __init__(self, emb_dim):
        super(GloveEncoderRNN, self).__init__()
        self.embedding_dim = emb_dim
        self.hidden_size = 600
        self.embedding = create_emb_layer(emb_dim, False)
        self.gru = nn.GRU(emb_dim, self.hidden_size, num_layers=1, bidirectional=False)
        self.lstm = nn.LSTM(emb_dim, self.hidden_size, num_layers=2, dropout=0.3, bidirectional=True)
        self.device = DEVICE

    def forward(self, input, hidden=None):
        embedded = self.embedding(input)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        #output, hidden = self.gru(output, hidden)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        return output, hidden

    #def initHidden(self):
        #return (torch.randn(2, 1, self.hidden_size // 2), torch.randn(2, 1, self.hidden_size // 2))
        #return torch.zeros(1, 1, self.hidden_size, device=self.device)
        #return torch.zeros(4, 1, self.hidden_size // 4)
        
class GloveAttnDecoderRNN(nn.Module):
    def __init__(self, emb_dim, dropout_p=0.1, max_length=MAX_LENGTH):
        super(GloveAttnDecoderRNN, self).__init__()
        self.embedding_dim = emb_dim
        self.hidden_size = 600
        self.output_size = MAPPER.n_words
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = create_emb_layer(emb_dim, False)
        #self.attn = nn.Linear(self.hidden_size + self.embedding_dim, self.max_length)
        #self.attn_combine = nn.Linear(self.hidden_size + self.embedding_dim, self.hidden_size)
        self.attn = Attn('concat', self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        #self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=2, dropout=0.3, bidirectional=True)
        self.concat = nn.Linear(self.hidden_size * 3, self.hidden_size)
        
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        rnn_output, hidden = self.lstm(embedded, last_hidden)
        #last_hidden = tuple of (num_layers * num_directions, batch, hidden_size)
        # ->((4, 1, 600), (4, 1, 600))
        #rnn_output dim : (seq_len=1, batch, num_directions*hidden_size)
        #encoder_outputs : (seq_len, batch, hidden_size)

        '''
        output = self.linearTest(rnn_output)
        output = self.out(output)
        return output, hidden, None
        '''

        attn_weights = self.attn(rnn_output, encoder_outputs)

        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        output = self.out(concat_output)
        output = F.log_softmax(output, dim=1)
	
        '''
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        #output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        '''
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 3, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        '''
        print "energy!!"
        print energy
        print energy.size()
        print torch.sum(self.v * energy, dim = 2).size()
        print encoder_output.size()
        '''
        return torch.sum(energy, dim=2)
        #return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
            
        attn_energies = attn_energies.t()
        
        '''
        print "attention start!"
        print attn_energies

        print F.softmax(attn_energies, dim=1)
        print F.softmax(attn_energies, dim=1).size()
        '''

        return F.softmax(attn_energies, dim=1).unsqueeze(1)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    #encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    #encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)
    loss = 0

    '''
    if input_length > max_length:
        input_length = max_length

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    '''
    encoder_outputs, (encoder_hidden, encoder_cell_state) = encoder(input_tensor)

    decoder_input = torch.tensor([[SOS_TOKEN]], device=DEVICE)

    decoder_hidden = encoder_hidden
    decoder_cell_state = encoder_cell_state

    use_teacher_forcing = True if random.random() < TFR else False

    if(target_length > MAX_LENGTH):
        target_length = MAX_LENGTH

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, (decoder_hidden, decoder_cell_state), decoder_attention = decoder(decoder_input, (decoder_hidden, decoder_cell_state), encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, (decoder_hidden, decoder_cell_state), decoder_attention = decoder(decoder_input, (decoder_hidden, decoder_cell_state), encoder_outputs)
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

def trainIters(encoder, decoder, n_iters, start_index, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0    

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = []
    for i in range(n_iters):
        training_pairs.append(MAPPER.tensorsFromPair(PAIRS[i + start_index]))
    #training_pairs = [MAPPER.tensorsFromPair(random.choice(PAIRS)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter-1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
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

def evaluate(encoder, decoder, paragraph, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = MAPPER.tensorFromParagraph(paragraph)
        input_length = input_tensor.size()[0]

        encoder_outputs, (encoder_hidden, encoder_cell_state) = encoder(input_tensor)

        decoder_input = torch.tensor([[SOS_TOKEN]], device=DEVICE)

        decoder_hidden = encoder_hidden
        decoder_cell_state = encoder_cell_state

        decoded_words = []
        
        for di in range(max_length):
            decoder_output, (decoder_hidden, decoder_cell_state), decoder_attention = decoder(decoder_input, (decoder_hidden, decoder_cell_state), encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_TOKEN:
                break
            else:
                decoded_words.append(MAPPER.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
        return decoded_words

def evaluatePairs(encoder, decoder, start, n=100):
    bleu_scores = []
    for i in range(n):
        pair = PAIRS[60000 + i + start]
        print('T: ' + pair[0])
        print('Q: ' + pair[1])

        output_words = evaluate(encoder, decoder, pair[0])
        output_question = ' '.join(output_words)
        print('Q? ' + output_question)
        #print(attentions)
        bleu_score = sentence_bleu([MAPPER.normalizeString(pair[1])], output_question)
        print("BLEU score: %f" % bleu_score)
        bleu_scores.append(bleu_score)
        print(' ')

    print("BLEU score average: %f" % (sum(bleu_scores) / float(len(bleu_scores))))

def model_train_test(n_iters, start_index, print_every):
    emb_dim = 200
    mapper = WordIndexMapper("word_to_index.pkl", "index_to_word.pkl", "word_to_count.pkl")
    #encoder1 = EncoderRNN(mapper.n_words, hidden_size).to(device)
    #decoder1 = DecoderRNN(hidden_size, mapper.n_words).to(device)
    encoder = GloveEncoderRNN(emb_dim).to(DEVICE)
    #attn_decoder1 = AttnDecoderRNN(hidden_size, mapper.n_words).to(device)
    decoder = GloveAttnDecoderRNN(emb_dim).to(DEVICE)
    trainIters(encoder, decoder, n_iters, start_index, print_every, plot_every=1000)
    evaluatePairs(encoder, decoder, start_index)

def load_models(PATH):
    hidden_size = 256
    mapper = WordIndexMapper("word_to_index.pkl", "index_to_word.pkl", "word_to_count.pkl")
    encoder = EncoderRNN(mapper.n_words, hidden_size).to(DEVICE)
    attn_decoder = AttnDecoderRNN(hidden_size, mapper.n_words).to(device)
    encoder.load_state_dict(torch.load(PATH + "50k_encoder"))
    attn_decoder.load_state_dict(torch.load(PATH + "50k_decoder"))
    evaluateRandomly(encoder, attn_decoder, mapper, True)
    
