from dict_generator import WordIndexMapper

reset_maker = WordIndexMapper("word_to_index.pkl", "index_to_word.pkl", "word_to_count.pkl")

reset_maker.reset()

print(reset_maker.normalizeString("Hey whats up"))
