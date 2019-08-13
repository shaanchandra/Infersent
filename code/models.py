#models.py

# #This is the linear CLassifier network. Used stand-alone for baseline, and used inside LSTM_main for the other LSTM models
# class Classifier(nn.Module):
#     def __init__(self,config):
#         super(Classifier, self).__init__()
#         #Initializing parameters and network architecture
#         self.bsize = config['b_size']
#         self.word_emb_dim = config['emb_dim']
#         self.fc_dim = config['fc_dim']
        
#         if config['model_name'] == 'base':
#             self.input_dim = self.word_emb_dim*4
#         elif config['model_name'] == 'lstm':
#             self.input_dim = config['lstm_dim']*4
#         elif config['model_name'] == 'bilstm' or config['model_name'] == 'bilstm_pool':
#             self.input_dim = config['lstm_dim']*4*2
        
#         self.n_classes = config['n_classes']
        
#         self.net = nn.Sequential(nn.Linear(self.input_dim, self.fc_dim), nn.Linear(self.fc_dim, self.n_classes))
        
#     def forward(self, features):
#         out = self.net(features)
#         return out


# #The main LSTM Class for all LSTM-type networks. Includes instance of LSTM/biLSTM/biLSTM(with pool) based on option and the Classifier      
# class LSTM_main(nn.Module):
#     def __init__(self, config):
#         super(LSTM_main, self).__init__()
        
#         #Initializing parameters and network architecture
#         self.bsize = config['b_size']
#         self.word_emb_dim = config['emb_dim']
#         self.lstm_dim = config['lstm_dim']

#         self.classif = Classifier(config).to(device)
        
#         #Based on model_name, instantiate the right LSTM class
#         if config['model_name'] == 'lstm':
#             self.encoder = LSTM(config).to(device)
#         else:
#             self.encoder = biLSTM(config).to(device)
        
#     def forward(self, tuples):
        
#         s1_tuple, s2_tuple = tuples
        
#         #'u' and 'v' are encoded sentences of premise(s1) and hypothesis(s2)
#         u = self.encoder(s1_tuple)
#         v = self.encoder(s2_tuple)
        
#         #create the features using the method in paper
#         features = torch.cat((u, v, torch.abs(u- v), u*v), 1)
            
#         #Feed to classifier for predicitons
#         out = self.classif(features)
#         return out
    
    
        
# #This is the LSTM class for uni-LSTM type network training
# class LSTM(nn.Module):
#     def __init__(self, config):
#         super(LSTM, self).__init__()
#         #Initializing parameters and network architecture
        
#         self.bsize = config['b_size']
#         self.word_emb_dim = config['emb_dim']
#         self.lstm_dim = config['lstm_dim']      

#         self.lstm = nn.LSTM(self.word_emb_dim, self.lstm_dim, 1, bidirectional=False)

#     def forward(self, sent_tuple):
#         sent, sent_len = sent_tuple

#         # Sort by length (descending) and keep original idx
#         sent_len_sorted = np.sort(sent_len)[::-1].copy()
#         idx_sort = [index for index, num in sorted(enumerate(sent_len), reverse = True, key=lambda x: x[1])]
# #         sent = sent.index_select(1, torch.cuda.LongTensor(idx_sort))
#         sent = sent.index_select(1, torch.LongTensor(idx_sort).to(device))
    
#         # Handling padding in Recurrent Networks
#         sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
        
#         #nn.LSTM returns: (ALL, (h0, c0))
#         #where, ALL = all the hidden states, (h0, c0): cell state and hidden state of the last time-step
#         #we are interested in h0. Hence, using the following indexing we obtain it as our output
#         sent_output = self.lstm(sent_packed)[1][0].squeeze(0)  # batch x 2*nhid

#         # Un-sort by length (since the labels need to correspond to the right pairs)
#         idx_unsort = np.argsort(idx_sort)
#         emb = sent_output.index_select(0, torch.LongTensor(idx_unsort).to(device))
# #         emb = sent_output.index_select(0, torch.cuda.LongTensor(idx_unsort))
#         return emb
        
        
        
        
# #This is the LSTM class for bi-LSTM type network training        
# class biLSTM(nn.Module):
    
#     def __init__(self, config):
#         super(biLSTM, self).__init__()
#         #Initializing parameters and network architecture
        
#         self.bsize = config['b_size']
#         self.word_emb_dim = config['emb_dim']
#         self.lstm_dim = config['lstm_dim']
        
#         # Chosse whether to return concatenation(biLSTM) or max-pool(biLSTM max-pool) results
#         self.pool = 1 if config['model_name'] == 'bilstm_pool' else 0

#         self.bilstm = nn.LSTM(self.word_emb_dim, self.lstm_dim, 1,
#                                 bidirectional=True)

#     def forward(self, sent_tuple):

#         sent, sent_len = sent_tuple

#         # Sort by length (descending) and keep original idx

#         sent_len_sorted = np.sort(sent_len)[::-1].copy()
#         idx_sort = [index for index, num in sorted(enumerate(sent_len), reverse = True, key=lambda x: x[1])]
# #         sent = sent.index_select(1, torch.cuda.LongTensor(idx_sort))
#         sent = sent.index_select(1, torch.LongTensor(idx_sort).to(device))
    
#         # Handling padding in Recurrent Networks
#         sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
        
#         #bi-directional LSTM returns : (ALL, (h0, c0)) 
#         #where (h0,c0) have the last hidden and cell states of left-to-right and right-to-left LSTMs
#         #ALL is the concatenation of each hidden state at every time step in the l-t-r and r-t-l direction LSTMs
#         X, hn = self.bilstm(sent_packed)     
#         X = nn.utils.rnn.pad_packed_sequence(X)[0]
        
#         #If not max-pool biLSTM, we extract the h0_l and h0_r from the tuple of tuples 'hn', and concat them to get the final embedding
#         if self.pool == 0:
#             sent_output = torch.cat((hn[0][0], hn[0][1]), 1)  
#             # Un-sort by length
#             idx_unsort = np.argsort(idx_sort)
#             emb = sent_output.index_select(0, torch.LongTensor(idx_unsort).to(device))
# #             emb = sent_output.index_select(0, torch.cuda.LongTensor(idx_unsort))
        
#         #If it is max-pooling biLSTM, set the PADS to very low numbers so that they never get selected in max-pooling
#         #Then, max-pool over each dimension(which is now 2D, as 'X' = ALL) to get the final embedding
#         elif self.pool == 1:

#             sent_output = torch.where(X == 0, torch.tensor(-1e8), X)  #replace PADs with very low numbers so that they never get picked
#             sent_output = torch.max(sent_output, 0)[0]  # (B x 2D)
            
#             # Un-sort by length
#             idx_unsort = np.argsort(idx_sort)
# #             emb = sent_output.index_select(0, torch.cuda.LongTensor(idx_unsort))
#             emb = sent_output.index_select(0, torch.LongTensor(idx_unsort).to(device))

#         return emb


import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

class SNLI(nn.Module):
    def __init__(self, config, pretrained_vectors = None):
        super(SNLI, self).__init__()
        self.vocab_size = config['vocab_size']
        self.embed_dim = config['embed_dim']
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        if pretrained_vectors is not None:
            self.embedding.weight.data.copy_(pretrained_vectors)
        self.embedding.requires_grad = False
        self.fc_dim = config['fc_dim']
        self.num_classes = config['n_classes']


        if config['model_name'] == 'base':
            self.encoder = Baseline()
            self.lstm_dim = config['embed_dim']
        elif config['model_name'] == 'lstm':
            self.encoder = LSTM(self.embed_dim, self.lstm_dim)
            self.lstm_dim = config['lstm_dim']
        elif config['model_name'] == 'bilstm':
            self.encoder = BiLSTM(self.embed_dim, self.lstm_dim)
            self.lstm_dim = 2*config['lstm_dim']
        elif config['model_name'] == 'bilstm_pool':
            self.encoder = BiLSTM(self.embed_dim, self.lstm_dim, bi = True)
            self.lstm_dim = 2*config['lstm_dim']
        else:
            raise ValueError("[!] ERROR: The encoder name is not correct! Please choose one of base/lstm/bilst/bilstm_pool")

        self.net = nn.Sequential(nn.Linear(4*self.lstm_dim, self.fc_dim),
                                 nn.Tanh(),
                                 nn.Linear(self.fc_dim, self.num_classes))


    def forward(self, s1, s1_len, s2, s2_len):
        s1 = self.embedding(s1)
        s2 = self.embedding(s2)

        u = self.encoder(s1, s1_len)
        v = self.encoder(s2, s2_len)
        feat = torch.cat((u, v, torch.abs(u - v), u*v), dim=1)

        out = self.net(feat)
        return out



# Baseline is the average of all the word embeddings of a sentence
class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()

    def forward(self, embed, length):
        out = torch.sum(embed, dim=0) / length.unsqueeze(1).float()
        return out



class LSTM(nn.Module):
    def __init__(self, embed_dim, lstm_dim):
        super(LSTM, self).__init__()
        self.encoder = nn.LSTM(embed_dim, lstm_dim, bidirectional = False)

    def forward(self, embed, length):
        sorted_len, sorted_idxs = torch.sort(length, descending =True)
        embed = embed[ : , sorted_idxs, :]

        packed_embed = pack_padded_sequence(embed, sorted_len, batch_first = False)
        # nn.LSTM returns: (ALL, (h0, c0))
        # where, ALL = all the hidden states, (h0, c0): hidden state and cell state of the last time-step
        # we are interested in h0. Hence, using the following indexing we obtain it as our output
        final_state = self.encoder(packed_embed)[1][0].squeeze(0)
        _, unsorted_idxs = torch.sort(sorted_idxs)
        out = final_state[unsorted_idxs, :]
        return out



class BiLSTM(nn.Module):
    def __init__(self, embed_dim, lstm_dim, bi=False):
        super(BiLSTM, self).__init__()
        self.pool = bi
        self.encoder = nn.LSTM(embed_dim, lstm_dim, bidirectional = True)

    def forward(self, embed, length):
        sorted_len, sorted_idxs = torch.sort(length, descending =True)
        embed = embed[ : , sorted_idxs, :]

        packed_embed = pack_padded_sequence(embed, sorted_len, batch_first = False)
        all_states, hidden_states = self.encoder(packed_embed)
        all_states, _ = pad_packed_sequence(all_states, batch_first = False)

        # If not max-pool biLSTM, we extract the h0_l and h0_r from the tuple of tuples 'hn', and concat them to get the final embedding
        if not self.pool:
            out = torch.cat((hidden_states[0][0], hidden_states[0][1]))

        # If it is max-pooling biLSTM, set the PADS to very low numbers so that they never get selected in max-pooling
        # Then, max-pool over each dimension(which is now 2D, as 'X' = ALL) to get the final embedding
        elif self.pool:
            # replace PADs with very low numbers so that they never get picked
            out = torch.where(all_states == 0, torch.tensor(-1e8), all_states)
            out, _ = torch.max(out, 0)

        _, unsorted_idxs = torch.sort(sorted_idxs)
        out = out[unsorted_idxs, :]
        return out


