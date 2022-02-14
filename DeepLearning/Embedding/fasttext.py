from gensim import models
def load_fasttext():
    return models.fasttext.load_facebook_vectors('/home/kmy/PycharmProjects/cc.ko.300.bin')
def similar_word(model, word):
    return model.most_similar(word)
if __name__ =='__main__':
    ko_model=load_fasttext()
    result = similar_word(ko_model,"파이썬")
    print(result)
    print(result[0])
    print(result[0][0])