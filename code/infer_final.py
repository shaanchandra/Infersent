import os
import numpy as np
import torch
import torch.nn as nn




#models.py


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Classifier(nn.Module):
    def __init__(self,config):
        super(Classifier, self).__init__()
        self.bsize = config['b_size']
        self.word_emb_dim = config['emb_dim']
        self.fc_dim = config['fc_dim']
        
        if config['model_name'] == 'base':
            self.input_dim = self.word_emb_dim*4
        elif config['model_name'] == 'lstm':
            self.input_dim = config['lstm_dim']*4
        elif config['model_name'] == 'bilstm' or config['model_name'] == 'bilstm_pool':
            self.input_dim = config['lstm_dim']*4*2
        
        self.n_classes = config['n_classes']
        
        self.net = nn.Sequential(nn.Linear(self.input_dim, self.fc_dim), nn.Linear(self.fc_dim, self.n_classes))
        
    def forward(self, features):
        out = self.net(features)

        return out
        
        

class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.bsize = config['b_size']
        self.word_emb_dim = config['emb_dim']
        self.lstm_dim = config['lstm_dim']
        
#         self.pool_type = config['pool_type']
      

        self.lstm = nn.LSTM(self.word_emb_dim, self.lstm_dim, 1,
                                bidirectional=False)
#dropout=self.dpout)

    def forward(self, sent_tuple):
        # sent_len [max_len, ..., min_len] (batch)
        # sent (seqlen x batch x worddim)

        sent, sent_len = sent_tuple

        # Sort by length (keep idx)

        sent_len_sorted = np.sort(sent_len)[::-1].copy()
        idx_sort = [index for index, num in sorted(enumerate(sent_len), reverse = True, key=lambda x: x[1])]

        sent = sent.index_select(1, torch.cuda.LongTensor(idx_sort))
#         sent = sent.index_select(1, torch.LongTensor(idx_sort))
    
        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
        sent_output = self.lstm(sent_packed)[1][0].squeeze(0).to('cpu')  # batch x 2*nhid

              

        # Un-sort by length
        idx_unsort = np.argsort(idx_sort)
        emb = sent_output.index_select(0, torch.LongTensor(idx_unsort))
        #emb = sent_output.index_select(0, torch.cuda.LongTensor(idx_unsort))

        return emb
    
    

    
    
class LSTM_main(nn.Module):
    def __init__(self, config):
        super(LSTM_main, self).__init__()
        self.bsize = config['b_size']
        self.word_emb_dim = config['emb_dim']
        self.lstm_dim = config['lstm_dim']
#         self.pool_type = config['pool_type']
      
        self.classif = Classifier(config).to(device)
        if config['model_name'] == 'lstm':
            self.encoder = LSTM(config).to(device)
        else:
            self.encoder = biLSTM(config).to(device)
        
    def forward(self, tuples):
        s1_tuple, s2_tuple = tuples
        
        u = self.encoder(s1_tuple)
        v = self.encoder(s2_tuple)
        
        features = torch.cat((u, v, torch.abs(u- v), u*v), 1)
            
        out = self.classif(features)
        
        return out
        
        
        
        
        
class biLSTM(nn.Module):
    
    def __init__(self, config):
        super(biLSTM, self).__init__()
        self.bsize = config['b_size']
        self.word_emb_dim = config['emb_dim']
        self.lstm_dim = config['lstm_dim']
        self.pool = 1 if config['model_name'] == 'bilstm_pool' else 0
            
#         self.pool_type = config['pool_type']


        self.bilstm = nn.LSTM(self.word_emb_dim, self.lstm_dim, 1,
                                bidirectional=True)

    def forward(self, sent_tuple):
        # sent_len [max_len, ..., min_len] (batch)
        # sent (seqlen x batch x worddim)

        sent, sent_len = sent_tuple

        # Sort by length (keep idx)

        sent_len_sorted = np.sort(sent_len)[::-1].copy()
        idx_sort = [index for index, num in sorted(enumerate(sent_len), reverse = True, key=lambda x: x[1])]
        sent = sent.index_select(1, torch.cuda.LongTensor(idx_sort))
#         sent = sent.index_select(1, torch.LongTensor(idx_sort))
    
        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
        X, hn = self.bilstm(sent_packed)     # seqlen x batch x 2*nhid
        X = nn.utils.rnn.pad_packed_sequence(X)[0]
        #print(X.size())
        
        
        if self.pool == 0:
            sent_output = torch.cat((hn[0][0], hn[0][1]), 1)  # batch x 2*nhid
            # Un-sort by length
            idx_unsort = np.argsort(idx_sort)
#             emb = sent_output.index_select(0, torch.cuda.LongTensor(idx_unsort))
            emb = sent_output.index_select(0, torch.LongTensor(idx_unsort))
        
        elif self.pool == 1:

            sent_output = torch.where(X.to('cpu') == 0, torch.tensor(-1e8).to('cpu'), X.to('cpu'))  #replace PADs with very low numbers so that they never get picked
            sent_output = torch.max(sent_output, 0)[0]  #B,2D
            
            # Un-sort by length
            idx_unsort = np.argsort(idx_sort)
            #emb = sent_output.index_select(0, torch.cuda.LongTensor(idx_unsort))
            emb = sent_output.index_select(0, torch.LongTensor(idx_unsort))

        return emb