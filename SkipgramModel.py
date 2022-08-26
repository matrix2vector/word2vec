'''
Author: 王晓洁 867778117@qq.com
Date: 2022-08-21 20:34:18
LastEditors: 王晓洁 867778117@qq.com
LastEditTime: 2022-08-25 22:37:37
FilePath: /word2vec/SkipgramModel.py
Description: Skipgram Model
'''
import torch
import torch.nn as nn


class SkipgramModel(nn.Module):

    def __init__(self, vocabulary_size, embedding_dim, padding_idx=0, counter=None) -> None:
        super(SkipgramModel, self).__init__()
        if counter == None:
            self.counter = torch.tensor([1. / vocabulary_size for i in range(vocabulary_size)], dtype=torch.float32)
        else:
            self.counter = counter
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size,
                                      embedding_dim=embedding_dim,
                                      padding_idx=padding_idx)
        self.fc1 = nn.Linear(in_features=embedding_dim,
                             out_features=vocabulary_size)

    def forward(self, x_in, apply_softmax=False):
        x_context, y_target, neg_samples = x_in
        y_target = self.embedding(y_target)
        y_context = self.embedding(x_context)
        y = torch.matmul(y_context, y_target.transpose(2, 3)).exp().squeeze().sum(-1)
        neg_samples = self.embedding(neg_samples)
        fenmu =  torch.matmul(neg_samples, y_target.transpose(2,3)).squeeze().exp().sum(-1)
        prob_of_y_tar = torch.div(y ,fenmu) 
        return y_context.squeeze().sum(1).squeeze(), prob_of_y_tar.sum(-1)

