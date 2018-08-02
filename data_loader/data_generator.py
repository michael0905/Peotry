# -*-coding:utf-8-*-#
import numpy as np
import collections
from utils.utils import length, pad_sequence

class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        peotry_file = self.config.filename
        self.build_dataset(peotry_file)

    def next_batch(self, batch_size):
        batch_X = []
        idx = np.random.choice(len(self.poetry_vectors), batch_size)
        x = self.poetry_vectors[idx]
        batch_length = [len(p) for p in x]
        batch_X = np.array([pad_sequence(p, self.config.longest_length, self.dictionary[' ']) for p in x])
        batch_Y = np.copy(batch_X)
        batch_Y[:, :-1] = batch_X[:, 1:]
        yield batch_X, batch_Y, batch_length

    def get_word_num(self):
        return self.word_num

    def get_poetrys(self, poetry_file):
        poetrys = []
        with open(poetry_file, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                try:
                    title, content = line.strip().split(":")
                    content = content.replace(" ", "")
                    if '_' in content or '(' in content or \
                    '（' in content or '《' in content or \
                    '[' in content :
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
        poetrys = sorted(poetrys, key=lambda line: len(line))
        print("唐诗总数:", len(poetrys))
        # print("shortest poetry: {}".format(poetrys[0]))
        # print("longest poetry: {}".format(poetrys[-1]))
        print("示例：{}".format(poetrys[1000]))
        words = []
        for poetry in poetrys:
            words += [word for word in poetry]
        # print(words[:10])
        counter = collections.Counter(words)
        # 从大到小排序
        counter_pairs = sorted(counter.items(), key=lambda x: -x[1])
        # 从counter中解压，并获取当中的词(不重复)
        words, _ = zip(*counter_pairs)
        words = words[:len(words)] + (" ", )
        # word -> id
        dictionary = dict(zip(words, range(len(words))))
        # id -> word
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        print("dictionary size: {}".format(len(dictionary.keys())))

        poetry_vectors = [[dictionary[word] for word in poetry] for poetry in poetrys]
        print("poetry vector example: {}".format(poetry_vectors[1000]))
        # print("id 0: {}".format(reversed_dictionary[0]))
        self.word_num = len(dictionary.keys())
        print("word number: {}".format(self.word_num))
        self.dictionary = dictionary
        self.poetry_vectors = np.array(poetry_vectors)
        self.reversed_dictionary = reversed_dictionary
