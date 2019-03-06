from word_index_mapper import WordIndexMapper
import preprocess

#dict_generator.test()

mapper = WordIndexMapper("word_to_index.pkl", "index_to_word.pkl", "word_to_count.pkl")

test_tensor = mapper.tensorFromSentence("Here is my dog!")
print test_tensor
print test_tensor.size(0)
