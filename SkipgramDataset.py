'''
Author: 王晓洁 867778117@qq.com
Date: 2022-08-18 21:04:46
LastEditors: 王晓洁 867778117@qq.com
LastEditTime: 2022-08-25 22:36:35
FilePath: /word2vec/SkipgramDataset
Description: 
'''

import pandas
import torch
from pandas import DataFrame
from torch.utils.data import Dataset

from Tokenizer import Tokenizer
from Vocabulary import Vocabulary


class SkipgramDataset(Dataset):

    def __init__(self, dataframe: DataFrame,  tokenizer: Tokenizer, split='train') -> None:
        super().__init__()
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.split = split

    def set_split(self, split):
        assert(split in {'train', 'test', 'validation'})
        self.split = split

    def get_tokenizer(self):
        return self.tokenizer

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        #print(len(row.context.split(' ')))
        return {'x_data': self.tokenizer.tokenize([row.context]),
                'y_target': self.tokenizer.tokenize([row.target]),
                'neg_samples': self.neg_sampling(self.tokenizer, self.compute_counter(self.tokenizer), 10)}

    @classmethod
    def from_csv(cls, csv_file):
        dataframe = pandas.read_csv(csv_file)
        words_list = []
        for index, row in dataframe.iterrows():
            words_list = words_list + row.context.split(' ')
            words_list.append(row.target)
        words_set = words_list
        vocab = Vocabulary(words_set)
        tokenizer = Tokenizer(vocab=vocab)
        return cls(dataframe, tokenizer)

    def __len__(self):
        return len(self.dataframe)

    def neg_sampling(self, tokenizer, counter, n=10):
        if counter == None:
            counter = torch.tensor([1. / tokenizer.vocab.__len__() for i in range(tokenizer.vocab.__len__())],
                                   dtype=torch.float32)
        _, x_select_indics = self.pro_sampling(counter, n)
        return x_select_indics

    def pro_sampling(self, counter, n=10):
        prob = torch.rand(1, counter.size()[0])
        prob = prob * counter
        x_select, x_select_indics = torch.topk(prob, n)
        return x_select, x_select_indics

    def compute_counter(self, tokenizer):
        counter = torch.zeros(tokenizer.vocab.__len__(), dtype=torch.long)
        print(tokenizer.vocab.__len__())
        for index, row in self.dataframe.iterrows():
            counter[tokenizer.tokenize([row.target]).item()] += 1
        return counter
