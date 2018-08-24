import os
import os.path
import urllib.request
import zipfile
import collections
import pickle
import numpy as np
import keras
# import tensorflow as tf


def maybe_download(filename_path, url, expected_bytes):
    # Download a file if not present, and make sure it's the right size.
    filename = os.path.basename(filename_path)

    if not os.path.exists(filename_path):
        filename, _ = urllib.request.urlretrieve(url + filename, filename_path)
    statinfo = os.stat(filename_path)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename_path)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename_path + '. Can you get to it with a browser?')
    return filename_path

def read_data(filename):
    # Extract the first file enclosed in a zip file as a list of words.
    with zipfile.ZipFile(filename) as f:
        # data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        data = str(f.read(f.namelist()[0])).split()

    return data


def build_dataset(words, n_words):
    # Process raw inputs into a dataset.

    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def preprocess_data(window_size=3, vocab_size=10000, output_filename=None):
    #  ---------------------------
    #  ------- GET DATA ----------
    #  ---------------------------
    url = 'http://mattmahoney.net/dc/'
    file_path = '../data/text8.zip'
    filename = maybe_download(file_path, url, 31344016)

    corpus = read_data(file_path)

    # data - numeric index of each word in the corpus. Size = size of corpus.
    #        Words not in the vocabulary have an index 0
    # count - a list of tuples. A tuple contains a vocab word and its count in the corpus. Size = Size of chosen vocab.
    #         the list is sorted in descending order, i.e. the most common words are on top
    # dictionary - mapping from words to their numerical indeces.
    #              Indeces are sorted in descending order of word appearance. I.e. most common words have low indeces.
    # reverse_dictionary - mapping from numerical indeces to words

    data, count, dictionary, reverse_dictionary = build_dataset(corpus, n_words=vocab_size)

    #  ---------------------------


    #  ---------------------------
    #  ------- CREATE SKIPGRAM WINDOWS
    #  ---------------------------

    # creates a table with sampling probabilities according to Zipf law
    # more common words should be samples with lower probability and vice versa
    sampling_table = keras.preprocessing.sequence.make_sampling_table(vocab_size)

    couples, labels = keras.preprocessing.sequence.skipgrams(data, vocab_size,
                                                             window_size=window_size,
                                                             sampling_table=sampling_table,
                                                             seed=42)
    word_target, word_context = zip(*couples)
    word_target = np.array(word_target, dtype="int32")
    word_context = np.array(word_context, dtype="int32")

    #  ---------------------------
    #  ------- SAVE THE RELEVANT DATA
    #  ---------------------------
    relevant_objects = {
        'data': data,
        'count': count,
        'dictionary': dictionary,
        'reverse_dictionary': reverse_dictionary,
        'word_target': word_target,
        'word_context': word_context,
        'labels': labels
    }

    pickle.dump(relevant_objects, file=open(output_filename, 'wb'))
    return 0
