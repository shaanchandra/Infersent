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

MODEL_NAME_DEFAULT = 'bilstm_pool'
MODEL_PATH_DEFAULT = './checkout/pool_final.pickle'

DATA_PATH_DEFAULT = './data/snli'
VEC_PATH_DEFAULT = './data/glove/glove.840B.300d.txt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




s1_DEFAULT = 'A lady is in the park'
s2_DEFAULT = 'A girl is in the house'




parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type = str, default = MODEL_NAME_DEFAULT,
                      help='model name: base / lstm / bilstm / bilstm_pool')
parser.add_argument('--model_path', type = str, default = MODEL_PATH_DEFAULT,
                      help='path to saved models')
parser.add_argument('--snli_path', type = str, default = DATA_PATH_DEFAULT,
                      help='path to SNLI data to read and preprocess')
parser.add_argument('--vec_path', type = str, default = VEC_PATH_DEFAULT,
                      help='path to GLOVE')
parser.add_argument('--prem', type = str, default = s1_DEFAULT,
                      help='premise sentence')
parser.add_argument('--hyp', type = str, default = s2_DEFAULT,
                      help='hypothesis sentence')

FLAGS, unparsed = parser.parse_known_args()

base_path = './checkout/base_final.pickle'
lstm_path = './checkout/lstm_final.pickle'
pool_path = './checkout/pool_final.pickle'
bi_path = './checkout/bilstm_final.pickle'

from data import get_nli, build_vocab

nli_path = DATA_PATH_DEFAULT
glove_path = VEC_PATH_DEFAULT

train, dev, test = get_nli(nli_path)
vocab, embeddings = build_vocab(train['s1']+train['s2']+test['s1']+test['s2']+dev['s1']+dev['s2'], glove_path)

# print(type(FLAGS.prem))

def get_batch_from_idx(sent, word_emb, config):
   
    embedded_sents = np.zeros((len(sent), config['emb_dim']))

    for i in range(len(sent)):
        
        if config['model_name'] == 'base':
            #return batch embeddings of dimension (L x B x D)
            embedded_sents[i, :] = word_emb[sent[i]]/len(sent)

        else:
            embedded_sents[i, :] = word_emb[sent[i]]
                
    return torch.from_numpy(embedded_sents).float(), len(sent)



def main():

    #Print Flags
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

# main() 


config = {'model_name' : FLAGS.model_name,
         'emb_dim' : 300,
         'b_size' : 1,
         'fc_dim' : 512,
          'lstm_dim': 2048,
         'n_classes' : 3}




s1_embed, s1_len = get_batch_from_idx(FLAGS.prem.split(), embeddings, config)
s2_embed, s2_len = get_batch_from_idx(FLAGS.hyp.split(), embeddings, config)

models = ['base', 'lstm', 'bilstm', 'bilstm_pool']

print("\n\n Hi I am the   " + str(FLAGS.model_name) + "    model...!!")
print("\n Hhhhmmmmm....lemme think...\n")


if config['model_name']== 'base':
    model = Classifier(config).to(device)
    PATH = FLAGS.model_path
    model.load_state_dict(torch.load(PATH, map_location=device))
    model = model.to(device)
    
    u = torch.sum(s1_embed,0).to(device)
    v = torch.sum(s1_embed,0).to(device)

    
    feats = torch.cat((u, v, torch.abs(u- v), u*v), 0).to(device)
    
    with torch.no_grad():
            out = model.forward(feats).to(device)
            pred = torch.max(out,0)[1]
            
    
else: 
    if config['model_name'] == 'lstm':
        PATH = FLAGS.model_path
    elif config['model_name'] == 'bilstm':
        PATH = FLAGS.model_path
    elif config['model_name'] == 'bilstm_pool':
        PATH = FLAGS.model_path
    
    
    
    model = LSTM_main(config).to(device)
    
    model.load_state_dict(torch.load(PATH, map_location=device))
    model = model.to(device)

    s1_embed = s1_embed.expand(1,s1_len, -1).transpose(0,1)
    s2_embed = s2_embed.expand(1,s2_len, -1).transpose(0,1)
    
    s1_len = torch.as_tensor(s1_len, dtype=torch.int64).expand(1)
    s2_len = torch.as_tensor(s2_len, dtype=torch.int64).expand(1)
    

    with torch.no_grad():
            out = model.forward(((s1_embed, s1_len), (s2_embed, s2_len))).to(device)
            pred = torch.max(out[0],0)[1]
            
            
print("==========================================================================")

print("(encrypted) Model output:    ", str(out))

print("==========================================================================")
print("\nPremise: " + FLAGS.prem)
print("Hypothesis: " + FLAGS.hyp)
if pred == 0:
    print("\nPrediction is: entailment")
elif pred == 1:
    print("\nPrediction is: neutral")
elif pred == 2:
    print("\nPrediction is: contradiction")