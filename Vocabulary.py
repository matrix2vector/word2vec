'''
Author: 王晓洁 867778117@qq.com
Date: 2022-08-18 01:54:26
LastEditors: 王晓洁 867778117@qq.com
LastEditTime: 2022-08-25 22:35:52
FilePath: /word2vec/Vocabulary.py
Description: a class to demenstrate Vocabulary
'''
class Vocabulary(object):

    '''
    description: 
    param {*} self
    param {dict} word_to_index
    param {str} mask_word
    param {int} add_unk
    param {str} unk_word
    return {*}
    '''

    def __init__(self, words: set, list=None, mask_word: str = "<MASK.>", unk_word: str = "<UNK.>"):
        if words == None:
            self.word_to_index = {}
        else:
            words_set = set(words)
            self.word_to_index = {word: index for index,
                                  word in enumerate(words_set, 2)}

        self.index_to_word = {index: word for word,
                              index in self.word_to_index.items()}
        
    
        self.mask_word = mask_word
        self.unk_word = unk_word

        if self.mask_word not in self.word_to_index:
            self.word_to_index[self.mask_word] = 0
            self.index_to_word[0] = self.mask_word

        if self.unk_word not in self.word_to_index:
            self.word_to_index[self.unk_word] = 1
            self.index_to_word[1] = self.unk_word

    def to_serializable(self):
        return {'words'     : self.word_to_index.keys(),
                'mask_word' : self.mask_word,
                'unk_word'  : self.unk_word}

    @classmethod
    def from_serializable(cls, contents: dict):
        return cls(**contents)

    @classmethod
    def from_segmentated_corpurs(cls, segmentated_corpurs: list, delimiter=' '):
        words_list = []
        for seg in segmentated_corpurs:
            splited_seg = seg.split(delimiter)
            words_list = [words_list, splited_seg]
        words_set = set(words_list)
        word_to_index = {word: index for index, word in enumerate(words_set)}
        contents = {'words': word_to_index.keys()}
        return cls(**contents)

    def __len__(self):
        return len(self.word_to_index)

    def add_word(self, word: str) -> int:
        if str not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
            return index
        return self.word_to_index[word]

    def add_words(self, words: list) -> list:
        return [self.add_word(word) for word in words]

    def lookup_by_word(self, word: str) -> int:
        if word in self.word_to_index:
            return self.word_to_index[word]
        else:
            return self.word_to_index[self.unk_word]

    def lookup_by_index(self, index: int) -> str:
        if index in self.index_to_word:
            return self.index_to_word[index]
        else:
            return self.unk_word

    def remove_word(self, word: str) -> int:
        self.word_to_index.pop(word, -1)

    def remove_words(self, words: list) -> int:
        return [self.remove_word(word) for word in words]
