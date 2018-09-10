# -*- coding:utf-8 -*-

from sklearn.externals import joblib
import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataloader import TextClassDataLoader
from vocab import VocabBuilder, GloveVocabBuilder

model = torch.load(f='gen/rnn_50.ml', map_location='cpu')
print(model)

v_builder = GloveVocabBuilder(path_glove='glove/glove.6B.100d.txt')
d_word_index, embed = v_builder.get_word_index()


def generate_indexifyer(lst_text):
    indices = []
    for word in lst_text:
        if word in d_word_index:
            indices.append(d_word_index[word])
        else:
            indices.append(d_word_index['__UNK__'])
    return indices


def load(sentence):

    words = [x.lower() for x in sentence[0].split()]
    print('#sentence is: ')
    print(sentence)
    sentence = generate_indexifyer(words)
    sentence = tuple(zip(sentence, ))
    print('#sentence is: ')
    print(sentence)

    # get the length of each seq in your batch
    seq_lengths = torch.LongTensor(list(map(len, sentence)))
    # seq_lengths = torch.LongTensor([len(sentence), sentence])

    # dump padding everywhere, and place seqs on the left.
    # NOTE: you only need a tensor as big as your longest sequence
    seq_tensor = torch.zeros((len(sentence), seq_lengths.max())).long()
    for idx, (seq, seqlen) in enumerate(zip(sentence, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

    # SORT YOUR TENSORS BY LENGTH!
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    # seq_tensor = seq_tensor.transpose(0, 1)

    return seq_tensor, seq_lengths


def get_sentence():
    train_loader = TextClassDataLoader('data/input.txt', d_word_index, batch_size=1)
    arr = []
    for i, (seq, target, seq_lengths) in enumerate(train_loader):
        print(seq)
        print(target)
        print(seq_lengths)
        output = model(seq, seq_lengths)
        arr = output[0].data.numpy().tolist()
        print(arr)
        print(arr.index(max(arr)))
    return arr.index(max(arr))

# seq = 'Mouse journal related to consumption'
# seq_tensor = torch.LongTensor(train_loader.samples[0][1])
# print(seq_tensor)
# seq_lengths = torch.LongTensor(list(map(len, train_loader.samples[0][1])))
# print(seq_lengths)
#
# output = model(seq_tensor, seq_lengths)
# print(output)

    # print('Start predict...')
    # print(train_loader.samples)

    # seq, seq_lengths = load(['jie tang'])
    # print(seq)
    # print(seq_lengths)
    # output = model(seq, seq_lengths)
    # arr = output[0].data.numpy().tolist()
    # print(arr)
    # print(arr.index(max(arr)))
    # return arr.index(max(arr))


if __name__ == '__main__':
    # tag = get_sentence()
    # uniq_dic = {'intent': str(tag)}
    # print(uniq_dic)
    while(1):
        query = input()
        file = open('data/input.txt', "w", encoding="utf8")
        file.write('label\tbody\n')
        file.write(str(0) + '\t' + query + '\n')
        file.write(str(0) + '\t' + query + '\n')
        file.close()
        tag = get_sentence()
        uniq_dic = {'intent': str(tag)}
        print(uniq_dic)
