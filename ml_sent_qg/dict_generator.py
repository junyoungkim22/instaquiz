import squad_loader
import pickle

class WordIndexMapper:
    def __init__(self, w2i_filename, i2w_filename, w2c_filename):
        self.w2i_filename = w2i_filename
        self.i2w_filename = i2w_filename
        self.w2c_filename = w2c_filename
        w2i_pkl_file = open(w2i_filename, 'rb')
        self.word2index = pickle.load(w2i_pkl_file)
        w2i_pkl_file.close()
        i2w_pkl_file = open(i2w_filename, 'rb')
        self.index2word = pickle.load(i2w_pkl_file)
        i2w_pkl_file.close()
        w2c_pkl_file = open(w2c_filename, 'rb')
        self.word2count = pickle.load(w2c_pkl_file)
        w2c_pkl_file.close()
        self.n_words = len(self.word2index)

    def addParagraph(self, para):
        for sent in para.split('.'):
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


data = squad_loader.process_file("train-v2.0.json")


