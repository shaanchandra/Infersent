from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import argparse


import os
import sys
import time
import argparse

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_nli, build_vocab
from models import Classifier, LSTM, biLSTM, LSTM_main
from dev_test_evals import model_eval
from data import get_nli, build_vocab

MODEL_NAME_DEFAULT = 'bilstm_pool'


s1_DEFAULT = 'A lady is in the park'
s2_DEFAULT = 'A girl is in the house'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type = str, default = MODEL_NAME_DEFAULT,
                      help='model name: base / lstm / bilstm / bilstm_pool')
parser.add_argument('--prem', type = str, default = s1_DEFAULT,
                      help='premise')
parser.add_argument('--hyp', type = str, default = s2_DEFAULT,
                      help='hypothesis')

FLAGS, unparsed = parser.parse_known_args()

base_path = './checkout/base_final.pickle'
lstm_path = './checkout/lstm_final.pickle'
pool_path = './checkout/pool_final.pickle'
bi_path = './checkout/bilstm_final.pickle'

def get_batch_from_idx(sent, word_emb, config):
   
    embedded_sents = np.zeros((len(sent), config['emb_dim']))
    
    
    for i in range(len(sent)):
        
        if config['model_name'] == 'base':
            #return batch embeddings of dimension (L x B x D)
            embedded_sents[i, :] = word_emb[sent[i]]/len(sent)

        else:
            if sent[i] in word_emb:
                embedded_sents[i, :] = word_emb[sent[i]]
            else: embedded_sents[i, :] = np.zeros((config['emb_dim']))
                
    return torch.from_numpy(embedded_sents).float(), len(sent)

def main():

    #Print Flags
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

# main() 


nli_path = './data/snli'
glove_path = './data/glove/glove.840B.300d.txt'

train, dev, test = get_nli(nli_path)
vocab, embeddings = build_vocab(train['s1']+train['s2']+test['s1']+test['s2']+dev['s1']+dev['s2'], glove_path)

# print(type(FLAGS.prem))



s1 = test['s1']
s2 = test['s2']
labels = test['label']

config = {'model_name' : 'base',
         'emb_dim' : 300,
         'b_size' : 1,
         'fc_dim' : 512,
          'lstm_dim': 2048,
         'n_classes' : 3}

#bin1 = 1-5
#bin2 = 6-15
#bin3 = 16-40
#bin4 = 41 and more

bin1 = 0
bin2 = 0
bin3 = 0
bin4 = 0
tot1 = 0
tot2 = 0
tot3 = 0
tot4 = 0

# model = LSTM_main(config).to(device)
# PATH = bi_path

model = Classifier(config).to(device)
PATH = base_path

model.load_state_dict(torch.load(PATH, map_location=device))
model = model.to(device)
print(model)

for i in range(len(labels)):
    
    s1_embed, s1_len = get_batch_from_idx(s1[i].split(), embeddings, config)
    s2_embed, s2_len = get_batch_from_idx(s2[i].split(), embeddings, config)
    u = torch.sum(s1_embed,0).to(device)
    v = torch.sum(s1_embed,0).to(device)

    
    feats = torch.cat((u, v, torch.abs(u- v), u*v), 0).to(device)
    
    with torch.no_grad():
            out = model.forward(feats).to(device)
            pred = torch.max(out,0)[1]
    
    if label[i] == 0:
        tot1 += 1
        if label
    
    
#     s1_embed, s1_len = get_batch_from_idx(s1[i].split(), embeddings, config)
#     s2_embed, s2_len = get_batch_from_idx(s2[i].split(), embeddings, config)
    
#     s1_embed = s1_embed.expand(1,s1_len, -1).transpose(0,1)
#     s2_embed = s2_embed.expand(1,s2_len, -1).transpose(0,1)
    
#     s1_len = torch.as_tensor(s1_len, dtype=torch.int64).expand(1)
#     s2_len = torch.as_tensor(s2_len, dtype=torch.int64).expand(1)
    

#     with torch.no_grad():
#             out = model.forward(((s1_embed, s1_len), (s2_embed, s2_len))).to(device)
#             pred = torch.max(out[0],0)[1]
    
#     if s2_len > 1 and s2_len <=5:
#         tot1+=1
#         if pred == labels[i]:
#             bin1+=1
#     elif s2_len > 5 and s2_len <=15:
#         tot2+=1
#         if pred == labels[i]:
#             bin2+=1
#     elif s2_len > 15 and s2_len <=40:
#         tot3+=1
#         if pred == labels[i]:
#             bin3+=1
#     elif s2_len > 40:
#         tot4+=1
#         if pred == labels[i]:
#             bin4+=1



print("b1 = " ,bin1/tot1)
print("b2 = " ,bin2/tot2)
print("b3 = " ,bin3/tot3)
print("b4 = " ,bin4/tot4)