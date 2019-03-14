import glove_loader
glove_loader.process_glove_file()

dict = glove_loader.create_glove_vect_dict()
print(dict["the"])
