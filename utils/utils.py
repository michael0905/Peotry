import argparse
import numpy as np

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-m', '--mode',
        metavar='M',
        default='train',
        help='The Configuration file')
    argparser.add_argument(
        '-c', '--cell',
        metavar='C',
        default='lstm',
        help='The rnn cell type')
    args = argparser.parse_args()
    return args

def length(array):
    return len([x for x in array if x > 0])

def pad_sequence(sequence, max_length, pad):
    """
    pad or clip sequence to max_length
    sequence: sequence to be padded
    max_length: length of result
    pad: symbol to pad
    """
    padN = max(max_length - len(sequence), 0)
    result = sequence[:max_length - padN] + [pad] * padN
    return result

def probsToWord(weights, words):
    """probs to word"""
    prefixSum = np.cumsum(weights) #prefix sum
    ratio = np.random.rand(1)
    index = np.searchsorted(prefixSum, ratio * prefixSum[-1]) # large margin has high possibility to be sampled
    return words[index[0]]
