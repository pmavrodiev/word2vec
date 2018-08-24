import utils
import numpy as np

import os
import sys
import pickle

from keras.models import Model
from keras.layers import Input, Dense, Reshape, dot, Flatten
from keras.layers.embeddings import Embedding

import keras.backend as K

class SimilarityCallback:
    def __init__(self, rev_dict, valid_size, valid_examples, vocab_size, validation_model):
        self.reverse_dictionary = rev_dict
        self.valid_size = valid_size
        self.vocab_size = vocab_size
        self.valid_examples = valid_examples
        self.validation_model = validation_model

    def run_sim(self):
        for i in range(self.valid_size):
            valid_word = self.reverse_dictionary[self.valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(self.valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = self.reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    def _get_sim(self, valid_word_idx):
        sim = np.zeros((self.vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        for i in range(self.vocab_size):
            in_arr1[0, ] = valid_word_idx
            in_arr2[0, ] = i
            out = self.validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim


if __name__ == "__main__":

    vocabulary_size = 10000
    word2vec_input = "../data/input_word2vec.pkl"
    expected_bytes = 340671348

    if not os.path.exists(word2vec_input):
        relevant_objects_filename = utils.preprocess_data(vocab_size=vocabulary_size, output_filename=word2vec_input)
    statinfo = os.stat(word2vec_input)

    if statinfo.st_size == expected_bytes:
        print('Found and verified ', word2vec_input)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + word2vec_input + '. Can you generate it again?')
        sys.exit(1)

    relevant_objects = pickle.load(open(word2vec_input, 'rb'))
    reverse_dictionary = relevant_objects['reverse_dictionary']
    word_target = relevant_objects['word_target']
    word_context = relevant_objects['word_context']
    labels = relevant_objects['labels']

    #  ---------------------------
    #  ------- KERAS
    #  ---------------------------
    batch_size = 64


    input_target = Input(shape=(1,), batch_shape=(batch_size, 1))
    input_context = Input(shape=(1,), batch_shape=(batch_size, 1))

    embedding_size = 300
    embedding = Embedding(vocabulary_size, embedding_size, input_length=1, name='embedding')

    target = embedding(input_target)
    target = Reshape((embedding_size, 1))(target)

    context = embedding(input_context)
    context = Reshape((embedding_size, 1))(context)

    # setup a cosine similarity operation which will be output in a secondary model
    similarity = dot([target, context], axes=1, normalize=True)
    similarity = Flatten()(similarity)

    # now perform the dot product operation to get a similarity measure
    dot_product = dot([target, context], axes=1)
    dot_product = Flatten()(dot_product)
    # dot_product = Reshape((2,))(dot_product)
    # add the sigmoid output layer
    output = Dense(1, activation='sigmoid')(dot_product)

    # create the primary training model
    model = Model(input=[input_target, input_context], output=output)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    # create a secondary validation model to run our similarity checks during training
    validation_model = Model(input=[input_target, input_context], output=similarity)
    #  ---------------------------
    #  ------- WORD2VEC constants
    #  ---------------------------

    valid_size = 16  # random set of words to evaluate similarity on during training
    valid_window = 100  # only pick the most common words
    # this generates valid_size numbers between 1 and valid_window
    # I.e. 'valid_size' valid examples will be picked from the 'valid_window' most common words
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    sim_cb = SimilarityCallback(reverse_dictionary, valid_size, valid_examples, vocabulary_size, validation_model)

    #  ---------------------------
    #  ------- TRAINING LOOP
    #  ---------------------------
    epochs = 1000000

    arr_1 = np.zeros((batch_size, ))
    arr_2 = np.zeros((batch_size, ))
    arr_3 = np.zeros((batch_size, ))
    l = np.array(labels)
    for cnt in range(epochs):
        idx = np.random.randint(0, len(labels) - 1, batch_size)
        arr_1[0:batch_size, ] = word_target[idx]
        arr_2[0:batch_size, ] = word_context[idx]
        arr_3[0:batch_size, ] = l[idx]
        loss = model.train_on_batch([arr_1, arr_2], arr_3)
        if cnt % 100 == 0:
            print("Iteration {}, loss={}".format(cnt, loss))
        if cnt % 10000 == 0:
            sim_cb.run_sim()

    print("HOHO")
