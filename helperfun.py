'''
Author: 王晓洁 867778117@qq.com
Date: 2022-08-21 21:31:21
LastEditors: 王晓洁 867778117@qq.com
LastEditTime: 2022-08-25 22:33:39
FilePath: /word2vec/helper.py
Description: 
'''

import os
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from generate_batches import generate_batches
from SkipgramDataset import SkipgramDataset
from SkipgramModel import SkipgramModel


def make_train_state(args):
    return {
        'stop_early' : False,
        'early_stopping_step' : 0,
        'early_stopping_beat_val' : 1e8,
        'learning_rate' : args.learning_rate,
        'epoch_index' : 0,
        'train_loss' : [],
        'train_acc' : [],
        'model_filename' : args.model_state_file

        
    }


'''
description: hyperparameters and config setting
'''
args = Namespace(
    skipgram_csv=f"./test.csv",
    tokenizer_file=f"tokenizer.json",
    model_state_file="model.pth",
    save_dir="model_storage/ch5/skipgram",
    embedding_dim=250,
    batch_size=50,
    seed=147,
    num_epochs=100,
    early_stopping_criteria=5,
    catch_keyboard_interrupt=True,
    cuda=True,
    reload_from_file=True,
    expand_filepaths_to_save_dir=True,
    learning_rate = 1e-4,
)

if args.expand_filepaths_to_save_dir:
    args.tokenizer_file = os.path.join(args.save_dir, args.tokenizer_file)
    args.model_state_file = os.path.join(args.save_dir, args.model_state_file)

    print("Expanded filepaths:")
    print('\t{}'.format(args.tokenizer_file))
    print('\t{}'.format(args.model_state_file))



if torch.cuda.is_available():
    args.cuda=True
    args.device='cuda'
elif getattr(torch.backends,'mps') is not None:
    if torch.backends.mps.is_available():
        args.cuda=False
        args.device = 'mps'
else:
    args.cuda=False
    args.devoice= 'cpu'

print("Using cuda:{}".format(args.cuda))
print("Using {}".format(args.device))

def set_seed_everywhere(seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed=seed)
    if args.cuda:
        torch.cuda.manual_seed_all(seed=seed)


def handle_dirs(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


        

set_seed_everywhere(args.seed, args.cuda)
handle_dirs(args.save_dir)

if args.reload_from_file:
    print("Loading dataset and loading vectorizer")
    dataset = SkipgramDataset.from_csv(args.skipgram_csv)
else:
    print("Loading dataset and creating tokenizer")
    dataset = SkipgramDataset.load_dataset_and_make_tokenizer(args.skipgram_csv)
    dataset.save_tokenizer(args.tokenizer_file)
print(dataset.dataframe.__len__)
tokenizer = dataset.get_tokenizer()

model = SkipgramModel(vocabulary_size=len(tokenizer.vocab),embedding_dim=args.embedding_dim)

model = model.to(device=args.device)

loss_func = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=1)

train_state = make_train_state(args=args)

def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    ncorrect = torch.eq(y_pred_indices, y_target)
    return ncorrect / len(y_pred_indices) * 100


try:
    for epoch_index in range(args.num_epochs):
        train_state['epoch_index'] = epoch_index
        dataset.set_split('train')
        batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device)
        running_loss = 0.
        running_acc = 0.
        model.train()

        
        for batch_index, batch_dict in enumerate(batch_generator):
            optimizer.zero_grad()

            y_pred, y = model(x_in=[batch_dict['x_data'], batch_dict['y_target'], batch_dict['neg_samples']])
            # print("1:", y_pred.size())
            # print("2:", y.size()) 
            # print("3:", batch_dict['y_target'].size())
            loss = loss_func(y_pred, batch_dict['y_target'].squeeze())
            loss_t = loss.item()

            loss.backward()
            optimizer.step()
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)



        train_state['train_loss'].append(running_loss)
        train_state['train_acc'].append(running_acc)


except KeyboardInterrupt:
    print("Exiting loop")


def pretty_print(results):
    for item in results:
        print("...[%.2f] - %s"%(item[1], item[0]))

def get_cloest(target_word, word_to_index, embeddings, n=5):
    word_embedding = embeddings[word_to_index[target_word]]
    distances = []
    for word, index in word_to_index.items():
        if word == "<MASK>" or word == target_word:
            continue
        distances.append((word, torch.dist(word_embedding, embeddings[index])))
    results = sorted(distances, key=lambda x: x[1])[1:n+2]
    return results

    