# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import, division, unicode_literals

import time
import argparse
import sys
import io
import os
import numpy as np
import torch
from torch.autograd import Variable
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set PATHs
# path to the NLP datasets 
PATH_TO_DATA_DAFAULT = './SentEval/data'
# path to glove embeddings
PATH_TO_VEC_DEFAULT = './data/glove/glove.840B.300d.txt'
# path to senteval
PATH_TO_SENTEVAL_DEFAULT = './SentEval'

MODEL_NAME_DEFAULT = 'bilstm_pool'
PATH_TO_MODEL_DEFAULT = './checkout/pool_final.pickle'

base_path = './checkout/base_final.pickle'
lstm_path = './checkout/lstm_final.pickle'
pool_path = './checkout/pool_final.pickle'
bi_path = './checkout/bilstm_final.pickle'

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type = str, default = MODEL_NAME_DEFAULT,
                      help='model name: base / lstm / bilstm / bilstm_pool')
parser.add_argument('--senteval_path', type = str, default = PATH_TO_SENTEVAL_DEFAULT,
                      help='path to cloned senteval direc')
parser.add_argument('--data_path', type = str, default = PATH_TO_DATA_DAFAULT,
                      help='path to data in SentEval')
parser.add_argument('--model_path', type = str, default = PATH_TO_MODEL_DEFAULT,
                      help='path to saved models')
parser.add_argument('--vec_path', type = str, default = PATH_TO_VEC_DEFAULT,
                      help='path for Glove embeddings (850B, 300D)')

FLAGS, unparsed = parser.parse_known_args()

assert os.path.isfile(FLAGS.senteval_path) and os.path.isfile(FLAGS.data_path) and os.path.isfile(FLAGS.model_path) and os.path.isfile(FLAGS.vec_path),   'Set MODEL, SentEval and/or GloVe PATHs correctly!!'

from infer_final import LSTM_main, Classifier, LSTM, biLSTM


# import SentEval
sys.path.insert(0, FLAGS.senteval_path)
import senteval




# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            if isinstance(word, bytes):
                word = word.decode('UTF-8')
                
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 5
    words['</s>'] = 1e9 + 4
    words['<p>'] = 1e9 + 3
    words['unk'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(word2id):
    word_vec = {}
    path_to_vec = FLAGS.vec_path

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec



# SentEval prepare and batcher
def prepare(params, samples):
    _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(params.word2id)   
        
    if params['config']['model_name'] != 'base':
        if params['config']['model_name'] == 'lstm':
            PATH = FLAGS.model_path      

        elif params['config']['model_name'] == 'bilstm':  
            PATH = FLAGS.model_path

        elif params['config']['model_name']== 'bilstm_pool':
            PATH = FLAGS.model_path

        params['model'] = LSTM_main(params['config']).to(device)
        params['model'].load_state_dict(torch.load(PATH, map_location=device))
        params['model'] = params['model'].encoder.to(device)
        print(params['model'])
        
    return

def embed_batch(batch, params):
    
    sen_lens = np.array([len(sent) for sent in batch])
    max_len = np.max(sen_lens)
    embedded_sents = np.zeros((max_len, len(batch), params['config']['emb_dim']))
    for i in range(len(batch)):
        for j in range(len(batch[i])):
            if isinstance(batch[i][j], bytes):
                batch[i][j] = batch[i][j].decode('UTF-8')
            else: batch[i][j]=batch[i][j]
                
            if params['config']['model_name'] == 'base':
                #return batch embeddings of dimension (L x B x D)
                if batch[i][j] in params.word_vec:
                    embedded_sents[j, i, :] = params.word_vec[batch[i][j]]/sen_lens[i]

            else:

                if batch[i][j] in params.word_vec:
                    embedded_sents[j, i, :] = params.word_vec[batch[i][j]]
                else:
                    embedded_sents[j, i, :] = params.word_vec['unk']

    return torch.from_numpy(embedded_sents).float(), sen_lens





def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    
    if params['config']['model_name']== 'base':
        padded_batch, _ = embed_batch(batch, params)
        embeddings = torch.sum(padded_batch, 0).to('cpu')
    else:
        embs, sen_lens = embed_batch(batch, params) 
        batch = Variable(torch.tensor(embs).to(device)) 
        with torch.no_grad():
            
            embeddings = params['model'].forward((batch, sen_lens))
            embeddings.to('cpu')

    return embeddings



# Set params for SentEval
params_senteval = {'task_path': FLAGS.data_path, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}
params_senteval['config'] = {
    'emb_dim'       :  300   ,
    'lstm_dim'       :  2048   ,
    'fc_dim'         :  512         ,
    'b_size'          :  64    ,
    'n_classes'      :  3      ,
    'model_name'     :  FLAGS.model_name   ,
    }



# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transfer_tasks = ['STS14','MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'ImageCaptionRetrieval']
    results = se.eval(transfer_tasks)
    print(results)
