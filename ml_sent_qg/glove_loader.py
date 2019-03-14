import pickle
import bcolz
import numpy as np
from tqdm import tqdm

glove_path = 'glove'
glove_dim = '200'


def process_glove_file():
    vectors = bcolz.carray(np.zeros(1), rootdir=glove_path +  '/6B.' + glove_dim + '.dat', mode='w')
    words = []
    idx = 0
    word2idx = {}
    with open(glove_path +  '/glove.6B.' + glove_dim + 'd.txt', 'rb') as f:
        i = 0
        for l in tqdm(f):
            line = l.split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
            i += 1
            if i == 400000:
                break

        
    vectors = bcolz.carray(vectors[1:].reshape((400000, int(glove_dim))), rootdir=glove_path + '/6B.' + glove_dim + '.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(glove_path +  '/6B.' + glove_dim + '_words.pkl', 'wb'))
    pickle.dump(word2idx, open(glove_path +  '/6B.' + glove_dim + '_idx.pkl', 'wb'))

def create_glove_vect_dict():
    vectors = bcolz.open(glove_path + '/6B.' + glove_dim + '.dat')[:]
    words = pickle.load(open(glove_path +  '/6B.' + glove_dim + '_words.pkl', 'rb'))
    word2idx = pickle.load(open(glove_path +  '/6B.' + glove_dim + '_idx.pkl', 'rb'))
    return {w: vectors[word2idx[w]] for w in words}
