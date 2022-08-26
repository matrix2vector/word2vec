'''
Author: matrix2vector 867778117@qq.com
Date: 2022-08-18 22:06:56
LastEditors: 王晓洁 867778117@qq.com
LastEditTime: 2022-08-26 23:44:30
FilePath: /word2vec/preprocess
Description: preprocess the novel fils in JinYongNovels dir and apply sliding window to generate DataFrame
'''
import logging
import os

import jieba
import pandas
from pandas import DataFrame
import datetime
import logging
from tqdm import tqdm

'''
description: 
param {list} novel_paths : each novel file's path
param {*} window_radius : sliding window radius(half of context nums), so that each training data contains (2*window_radius+1) words
param {*} out_file_path : the file to storage returned DataFrame
param {*} delimiter : the delimiter amoung context
param {*} mask : <MASK>
return {*} : a DataFrame
'''
def main(novel_paths: list, window_radius, out_file_path, delimiter=' ', mask='<MASK>'):
    assert(window_radius >= 1)
    dataframe = DataFrame(columns=['context', 'target'])
    for novel_path in novel_paths:
        time_start = datetime.datetime.now()
        with open(novel_path, 'rb') as file:
            novel = file.read()
            novel_split = novel.split()
            l = len(novel_split)
            for i in tqdm(range(l)):
                seg = novel_split[i]
            # for seg in novel_split:
                seg = jieba.cut(seg)
                seg = delimiter.join(seg).split(delimiter)
                length = len(seg)
                mask_list = [mask for i in range(window_radius)]
                seg_masked = mask_list + seg + mask_list
                for i in range(window_radius, length+window_radius):
                    left_context = [seg_masked[i+j]for j in range(-window_radius, 0)]
                    right_context = [seg_masked[i+k]for k in range(1, window_radius+1)]
                    context = delimiter.join(left_context)+delimiter+delimiter.join(right_context)
                    target = seg_masked[i]
                    append_row = pandas.DataFrame([[context, target]],
                                                    columns=['context', 'target'])
                    dataframe = pandas.concat([dataframe, append_row],
                                                sort=False,
                                                ignore_index=True)
        time_end = datetime.datetime.now()
        print(novel_path, "use", time_end - time_start)
    with open(out_file_path, 'w'):
        dataframe.to_csv(out_file_path, encoding='utf-8')
    return dataframe


'''
description: get all the file names in dir
param {*} dir : directory path
return {*} : a list that consists of all the file path in dir
'''
def get_files_path(dir):
    paths = os.listdir(dir)
    return [os.path.join(dir, path) for path in paths]


novel_paths = get_files_path('./JinYongNovels')

if __name__ == "__main__":
    jieba.enable_parallel()
    jieba.setLogLevel(logging.CRITICAL)
    

    dataframe = main(novel_paths, 
                     window_radius=2,
                     out_file_path='./test.csv')
