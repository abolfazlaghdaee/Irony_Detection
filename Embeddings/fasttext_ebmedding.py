# !wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fa.300.vec.gz
# !gunzip cc.fa.300.vec.gz
# !pip install numpy==1.23.5

ft_model = KeyedVectors.load_word2vec_format('cc.fa.300.vec', binary=False)


from gensim.models import KeyedVectors
import numpy as np




def get_sentence_vector_fasttext(sentence):
    words = sentence.split()
    vectors = [ft_model[word] for word in words if word in ft_model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(ft_model.vector_size)








