# -*- coding:utf-8 -*-

from sklearn.externals import joblib
import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataloader import TextClassDataLoader
from vocab import VocabBuilder, GloveVocabBuilder

model = torch.load('gen/rnn_50.ml', map_location='cpu')
print(model)

v_builder = GloveVocabBuilder(path_glove='glove/glove.6B.100d.txt')
d_word_index, embed = v_builder.get_word_index()

def get_sentence():
    train_loader = TextClassDataLoader('data/input.txt', d_word_index, batch_size=1)

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
    for i, (seq, target, seq_lengths) in enumerate(train_loader):
        output = model(seq, seq_lengths)
        arr = output[0].data.numpy().tolist()
        print(arr)
        print(arr.index(max(arr)))
    return arr.index(max(arr))