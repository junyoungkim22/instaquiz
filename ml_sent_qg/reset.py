from word_index_mapper import WordIndexMapper
from word_index_mapper import make_dictionary

reset_maker = WordIndexMapper("word_to_index.pkl", "index_to_word.pkl", "word_to_count.pkl")

reset_maker.reset()

make_dictionary()
