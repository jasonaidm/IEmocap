from __future__ import division, print_function, absolute_import
import collections, os
import numpy as np
import string


def build_dataset(words):
    vocabulary_size = 20000
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
        unk_count = 0
    count[0][1] = unk_count
    #print(dictionary)
    return dictionary


def save_dictionary(path, dic):
    f = open(path, 'a')
    for key, value in dic.items():
        f.write(str(key)+' '+str(value)+'\n')
    f.close()


def onehot_vector(input_path, output_path):
    words = []
    f = open(input_path, 'r')
    for line in f:
        sentence = line.translate(None, string.punctuation)
        for word in sentence.split():
            words.append(word)
    f.close()
    dic = build_dataset(words)
    save_dictionary(output_path, dic)
    return dic

def get_dictionary(path):
    dic = dict()
    f = open(path, 'r')
    for line in f:
        tmp = []
        for word in line.split():
            tmp.append(word)
        dic[tmp[0]] = int(tmp[1])
    f.close()
    return dic


def embed_onehot(dictionary, data):
    result = []
    for word in data.split():
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
        result.append(index)
    return result


def initial_embed(dic, path):
    embeddings_index = {}
    f = open(os.path.join(path, 'glove.6B.200d.txt'), encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(dic) + 1, 200))
    for word, i in dic.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # em_text = Embedding(len(dic) + 1, 200, weights=[embedding_matrix], trainable=True)(text_input)

    return embedding_matrix
