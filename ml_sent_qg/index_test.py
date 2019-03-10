from word_index_mapper import WordIndexMapper
import preprocess
import random
import squad_loader

#dict_generator.test()

mapper = WordIndexMapper("word_to_index.pkl", "index_to_word.pkl", "word_to_count.pkl")
data = squad_loader.process_file("train-v2.0.json")
pairs = squad_loader.prepare_ans_tagged_pairs(data)

for i in range(10):
    pair = random.choice(pairs)
    context, question = pair
    print(context)
    print("*"*80)
    test_tensor = mapper.tensorFromSentence(context)
    print(test_tensor)
    print("&"*80)


test_tensor = mapper.tensorFromSentence("Here is my dog!")
print test_tensor
print test_tensor.size(0)
