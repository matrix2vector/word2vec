'''
Author: 王晓洁 867778117@qq.com
Date: 2022-08-21 19:59:29
LastEditors: 王晓洁 867778117@qq.com
LastEditTime: 2022-08-25 22:32:09
FilePath: /word2vec/generate_batches.py
Description: encapsulate DataLoader to make sure that all batches are on same device
'''
from torch.utils.data import DataLoader

'''
description: generate batches and garantee all batches are on same device
param {*} dataset
param {*} batch_size: batch size
param {*} shuffle: if apply shuffle
param {*} drop_last: if to drop off the last few data that less that batch_size
param {*} device: run on gpu or cpu or mps
return {*}: batches
'''
def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
    assert(device in {'cpu', 'cuda', 'mps'})
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    for data_dict in dataloader:
        output_data_dict = {}
        for name, tensor in data_dict.items():
            output_data_dict[name] = data_dict[name].to(device)
        yield output_data_dict