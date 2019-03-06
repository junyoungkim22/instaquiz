import squad_loader
import pickle
import re
import unicodedata

dict_directory = "dictionary/"

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
        pickle.dump({"SOS": 0, "EOS": 1}, output)
        output.close()
        output = open(self.i2w_filename, 'wb')
        pickle.dump({0: "SOS","EOS" : 1}, output)
        output.close()
        output = open(self.w2c_filename, 'wb')
        pickle.dump({}, output)
        output.close()
        
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicode.category(c) != 'Mn'
        )

    def normalizeString(self, s):
        #s = self.unicodeToAscii(s.lower().strip())
        s = s.lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
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
                                
def test():
    data = squad_loader.process_file("train-v2.0.json") 
    i = 0 
    limit = 13500
    maker = WordIndexMapper("word_to_index.pkl", "index_to_word.pkl", "word_to_count.pkl") 
    for context, context_qas in data: 
        if i < limit:
            maker.addParagraph(context) 
            i += 1 
        else:
            maker.paragraph_test(context)
            i += 1
            if i == limit + 100:
                break
    print(len(maker.word2index))
    maker.save()
