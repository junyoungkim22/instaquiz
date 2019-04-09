import squad_loader
import torch
import pickle
import re
import unicodedata
import random
from txt_token import SOS_TOKEN, EOS_TOKEN

dict_directory = "dictionary/"

'''
SOS_TOKEN = 0
EOS_TOKEN = 1
ANSS_TOKEN = 2
ANSE_TOKEN = 3
'''

class WordIndexMapper:
    def __init__(self, w2i_filename, i2w_filename, w2c_filename):
        self.w2i_filename = dict_directory + w2i_filename
        self.i2w_filename = dict_directory + i2w_filename
        self.w2c_filename = dict_directory + w2c_filename
        w2i_pkl_file = open(self.w2i_filename, 'rb')
        self.word2index = pickle.load(w2i_pkl_file)
        w2i_pkl_file.close()
        i2w_pkl_file = open(self.i2w_filename, 'rb')
        self.index2word = pickle.load(i2w_pkl_file)
        i2w_pkl_file.close()
        w2c_pkl_file = open(self.w2c_filename, 'rb')
        self.word2count = pickle.load(w2c_pkl_file)
        w2c_pkl_file.close()
        self.n_words = len(self.word2index)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def addParagraph(self, para):
        paragraph = self.normalizeString(para)
        for sent in paragraph.split('.'):
            self.addSentence(sent)

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        print "Trimming words with less than " + str(min_count) + " occurrences"
        before = self.n_words;
        print before
        keep_words = []

        for i in range(self.n_words):
            try:
                if self.word2count[self.index2word[i]] >= min_count:
                    keep_words.append(self.index2word[i])
                else:
                    #print self.index2word[i]
                    pass
            except KeyError:
                print self.index2word[i] + " not included!"
        
        self.word2index = {"<sos>": 0, "<eos>": 1, "<anss>": 2, "<anse>": 3, "<unk>": 4}
        self.word2count = {}
        self.index2word = {0: "<sos>", 1: "<eos>", 2: "<anss>", 3: "<anse>", 4: "<unk>"}

        self.n_words = len(self.word2index);

        for word in keep_words:
            self.addWord(word)

        print "Before trim n_words: " + str(before)
        print "After trim n_words: " + str(self.n_words)


    def save(self):
        output = open(self.w2i_filename, 'wb')
        pickle.dump(self.word2index, output)
        output.close()
        output = open(self.i2w_filename, 'wb')
        pickle.dump(self.index2word, output)
        output.close()
        output = open(self.w2c_filename, 'wb')
        pickle.dump(self.word2count, output)
        output.close()

    def reset(self):
        output = open(self.w2i_filename, 'wb')
        pickle.dump({"<sos>": 0, "<eos>": 1, "<anss>": 2, "<anse>": 3, "<unk>": 4}, output)
        output.close()
        output = open(self.i2w_filename, 'wb')
        pickle.dump({0: "<sos>", 1: "<eos>", 2: "<anss>", 3: "<anse>", 4: "<unk>"}, output)
        output.close()
        output = open(self.w2c_filename, 'wb')
        pickle.dump({}, output)
        output.close()
        self.word2index = {"<sos>": 0, "<eos>": 1, "<anss>": 2, "<anse>": 3, "<unk>": 4}
        self.word2count = {}
        self.index2word = {0: "<sos>", 1: "<eos>", 2: "<anss>", 3: "<anse>", 4: "<unk>"}
        self.n_words = len(self.word2index);
        
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicode.category(c) != 'Mn'
        )

    def indexesFromParagraph(self, paragraph):
        ret = []
        for sent in paragraph.split('.'):
            ret += self.indexesFromSentence(sent)
        return ret

    def indexesFromSentence(self, sentence):
        ret = []
        sentence = self.normalizeString(sentence)
        for word in sentence.split(' '):
            try:
                ret.append(self.word2index[word])
            except KeyError:
                ret.append(self.word2index["<unk>"])
                continue
        return ret

    def padSentence(self, sentence, pad_length)
        indexes = self.indexesFromSentence(sentence)
        if(len(indexes) < pad_length)
        indexes += [PAD_TOKEN] * ((pad_length) - len(indexes))
        return indexes

    def tensorFromParagraph(self, paragraph):
        indexes = self.indexesFromParagraph(paragraph)
        indexes.append(EOS_TOKEN)
        return torch.tensor(indexes, dtype=torch.long, device=self.device).view(-1, 1)
        
    def tensorFromSentence(self, sentence):
        indexes = self.indexesFromSentence(sentence)
        indexes.append(EOS_TOKEN)
        return torch.tensor(indexes, dtype=torch.long, device=self.device).view(-1, 1)

    def tensorsFromPair(self, pair):
        input_tensor = self.tensorFromParagraph(pair[0])
        target_tensor = self.tensorFromSentence(pair[1])
        return (input_tensor, target_tensor)

    def normalizeString(self, s):
        #s = self.unicodeToAscii(s.lower().strip())
        s = s.lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?<>]+", r" ", s)
        return s

    def paragraph_test(self, para):
        word_count = 0
        hit_count = 0
        paragraph = self.normalizeString(para)
        for sent in paragraph.split(' '):
            (hc, wc) = self.sentence_test(sent)
            word_count += wc
            hit_count += hc
        print ("word count: ", word_count)
        print ("hit count: ", hit_count)
        print ("percentage: " + str((float(hit_count) / word_count) * 100) + "%") 
        
    def sentence_test(self, sent): 
        word_count = 0 
        hit_count = 0 
        for word in sent.split(' '): 
            word_count += 1 
            if word in self.word2index: 
                hit_count += 1 
            return (hit_count, word_count) 
                                
def make_dictionary(limit, trim_num):
    data = squad_loader.process_file("train-v2.0.json") 
    #print "shuffling"
    random.shuffle(data)
    i = 0 
    maker = WordIndexMapper("word_to_index.pkl", "index_to_word.pkl", "word_to_count.pkl") 
    maker.reset()
    #for context, context_qas in data: 
    for i in range(limit):
        context, context_qas = data[i]
        if i < limit:
            maker.addParagraph(context) 
            i += 1 
        else:
            #maker.paragraph_test(context)
            i += 1
            if i == limit + 300:
                break
    #print(len(maker.word2index))
    maker.trim(trim_num)
    maker.save()
