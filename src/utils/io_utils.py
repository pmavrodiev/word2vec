import json
import pickle
import os
import shutil

from os.path import join, abspath, isfile

encoding = "utf-8"


def save_json(filename, data):
    with open(filename, "w+", encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False)


def load_json(filename):
    with open(filename, "r", encoding=encoding) as data_file:
        return json.load(data_file)


def save_pickle(filename, data):
    with open(filename, "wb", encoding=encoding) as output_file:
        pickle.dump(data, output_file, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, "rb", encoding=encoding) as openfile:
        return pickle.load(openfile)


def create_dir_next_to_file(filename):
    dirname = join(abspath(join(filename, os.pardir)), "tmp")
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)
    return dirname


def delete_dir(dirname):
    shutil.rmtree(dirname)


def merge_files(dirname):
    result = dict()
    for dirpath, dirnames, filenames in os.walk(dirname):
        for filename in [filename for filename in filenames if isfile(join(dirpath, filename))]:
            filepath = join(dirpath, filename)
            json_data = load_json(filepath)
            result.update(json_data)
    return result
