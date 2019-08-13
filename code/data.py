import os
import numpy as np
import torch


def get_batch_from_idx(batch, word_emb, config):
    sen_lens = np.array([len(x) for x in batch])
    max_len = np.max(sen_lens)
    embedded_sents = np.zeros((max_len, len(batch), config['emb_dim']))
    
    
    for i in range(len(batch)):
        for j in range(len(batch[i])):
            if config['model_name'] == 'base':
                #return batch embeddings of dimension (L x B x D)
                embedded_sents[j, i, :] = word_emb[batch[i][j]]/sen_lens[i]

            else:
                
                embedded_sents[j, i, :] = word_emb[batch[i][j]]

    return torch.from_numpy(embedded_sents).float(), sen_lens



def get_nli(data_dir):
    s1, s2 = {}, {}
    label = {}
    tags = {'entailment': 0, 'neutral':1, 'contradiction':2}

    for dset in ['train', 'dev', 'test']:
        s1[dset], s2[dset], label[dset] = {}, {}, {}
        s1[dset]['path'] = os.path.join(data_dir, 's1.'+ dset)
        s2[dset]['path'] = os.path.join(data_dir, 's2.'+ dset)
        label[dset]['path'] = os.path.join(data_dir, 'labels.'+ dset)
        
        s1[dset]['sent'] = [line.rstrip() for line in open(s1[dset]['path'], 'r')]
        s2[dset]['sent'] = [line.rstrip() for line in open(s2[dset]['path'], 'r')]
        label[dset]['label'] = np.array([tags[line.rstrip('\n')] for line in open(label[dset]['path'], 'r')])
        
        print(str(len(label[dset]['label'])) + " instances extracted of " + str(dset))

    train = {'s1': s1['train']['sent'], 's2': s2['train']['sent'],
             'label': label['train']['label']}
    dev = {'s1': s1['dev']['sent'], 's2': s2['dev']['sent'],
           'label': label['dev']['label']}
    test = {'s1': s1['test']['sent'], 's2': s2['test']['sent'],
            'label': label['test']['label']}
    print("Example: train['s1'][0] = ", train['s1'][0])
    return train, dev, test



def build_vocab(lines, glove_path):
    word_dict = {}
    embedding = {}
    not_in_vocab = []
    print("\n -----------Building Vocab from the SNLI dataset-----------")
    for l in lines:
        for word in l.split():
            if word not in word_dict:
                word_dict[word] = ''
        
    #start sentence token
    word_dict['<s>'] = ''
    
    #end sentence token
    word_dict['</s>'] = ''
    
    #padding token for batching
    word_dict['<p>'] = ''
    
    print("----Getting Glove word embedding for each word in vocab (non vocab words ignored, ie, <unk> not used)----")          
    with open(glove_path, encoding="utf8") as f:
        for sents in f:
            word, emb = sents.split(' ',1)
            if word in word_dict:
                embedding[word] = np.fromstring(emb, sep=' ')
#                 np.array(list(map(float, emb.split())))

    print("Found "+ str(len(embedding)) + " words with Glove embeddings out of "+ str(len(word_dict)) + " total words in corpus.")
    return word_dict, embedding


class SNLIBatchGenerator():
    def __init__(self, bucket_iterator, premise_field = 'premise', hyp_field = 'hypothesis', label_field = 'label'):
        self.bucket_iterator = bucket_iterator
        self.premise_field = premise_field
        self.hyp_field = hyp_field
        self.label_field = label_field

    def __len__(self):
        return len(self.bucket_iterator)

    def __iter__(self):
        for batch in self.bucket_iterator:
            premise = getattr(batch, self.premise_field)
            hyp = getattr(batch, self.hyp_field)
            label = getattr(batch, self.label_field)

            yield premise, hyp, label