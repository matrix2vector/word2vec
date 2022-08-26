'''
Author: 王晓洁 867778117@qq.com
Date: 2022-08-18 19:03:22
LastEditors: 王晓洁 867778117@qq.com
LastEditTime: 2022-08-25 22:35:44
FilePath: /word2vec/Tokenizer.py
Description: Tokenizer
'''
import torch
import torch.nn as nn
from pandas import DataFrame

from Vocabulary import Vocabulary


class Tokenizer(object):

    def __init__(self, vocab: Vocabulary, pad=0) -> None:
        self.vocab = vocab
        self.pad = pad

    def tokenize(self, contents: str, delimiter=' ', len: int = -1):
        contents_split = contents.split(delimiter)

        return torch.tensor([self.vocab.lookup_by_word(word) for word in contents_split], dtype=torch.int64)

    def tokenize(self, contents_list: list, delimiter=' ', len: int = -1):
        return nn.utils.rnn.pad_sequence([torch.tensor([self.vocab.lookup_by_word(word) for word in contents.split(delimiter)], dtype=torch.int64) for contents in contents_list],
                                         batch_first=True, padding_value=self.pad)

    @classmethod
    def from_dataframe(cls, dataframe: DataFrame):
        for index, row in dataframe.iterrows():
            row.context

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def to_serializable(self) -> str:
        return {'vocab': self.vocab.to_serializable(),
                'pad': self.pad}
                