from dict_generator import WordIndexMapper
import preprocess

#dict_generator.test()

mapper = WordIndexMapper("word_to_index.pkl", "index_to_word.pkl", "word_to_count.pkl")

test_tensor = preprocess.tensorFromSentence(mapper, mapper.normalizeString("Here is my dog!"))
print test_tensor.size(0)
print mapper.index2word[2582]
