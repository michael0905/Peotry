# -*-coding:utf-8-*-#
import numpy as np
import collections
import random
from utils.utils import length, pad_sequence

class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        peotry_file = self.config.filename
        self.build_dataset(peotry_file)

    def next_batch(self, batch_size):
        batch_X = []
        batch_Y = []
        idx = np.random.choice(len(self.poetry_vectors), batch_size)
        x = self.poetry_vectors[idx]
        max_length = max([len(vector) for vector in x])
        batch_X = np.full((batch_size, max_length), self.dictionary[" "], np.int32) # padding space
        for j in range(batch_size):
            batch_X[j, :len(x[j])] = x[j]
        batch_Y = np.copy(batch_X)
        batch_Y[:, :-1] = batch_X[:, 1:]
        yield batch_X, batch_Y

    def get_word_num(self):
        return self.word_num

    def get_poetrys(self, poetry_file):
        poetrys = []
        with open(poetry_file, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                try:
                    title, author, content = line.strip().split("::")
                    content = content.replace(" ", "")
                    if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content :
                        continue
                    if len(content) < self.config.shortest_length or len(content) > self.config.longest_length:
                        continue
                    content = '[' + content + ']'
                    poetrys.append(content)
                except Exception as e:
                    pass
            return poetrys

    def build_dataset(self, filename):
        poetrys = self.get_poetrys(filename)
        word_freq = collections.Counter()
        for poetry in poetrys:
            word_freq.update(poetry)

        word_freq[" "] = -1
        word_pairs = sorted(word_freq.items(), key = lambda x: -x[1])
        self.words, freq = zip(*word_pairs)
        self.word_num = len(self.words)

        self.dictionary = dict(zip(self.words, range(self.word_num))) #word to ID
        poetry_vectors = [([self.dictionary[word] for word in poetry]) for poetry in poetrys] # poetry to vector
        self.poetry_vectors = np.array(poetry_vectors)
        self.reversed_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))

        print("唐诗总数: {}".format(len(self.poetry_vectors)))
        print("字典词数: {}".format(len(self.dictionary.keys())))
