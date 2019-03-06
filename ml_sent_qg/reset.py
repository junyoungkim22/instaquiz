from word_index_mapper import WordIndexMapper

reset_maker = WordIndexMapper("word_to_index.pkl", "index_to_word.pkl", "word_to_count.pkl")

reset_maker.reset()
