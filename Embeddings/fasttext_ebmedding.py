# !wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fa.300.vec.gz
# !gunzip cc.fa.300.vec.gz


from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm

ft_model = KeyedVectors.load_word2vec_format('cc.fa.300.vec', binary=False)

def get_sentence_vector_fasttext(sentence):
    words = sentence.split()
    vectors = [ft_model[word] for word in words if word in ft_model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(ft_model.vector_size)

def get_vectors_for_series(text_series):
    return np.vstack([get_sentence_vector_fasttext(sentence) for sentence in tqdm(text_series, desc="Embedding sentences")])





