# InferSent
Learning general-purpose sentence representations in the natural language inference (NLI) task.

Implement the InferSent model introduced [here](https://arxiv.org/abs/1705.02364) by Conneau et.al. 
NLI is the task of classifying entailment or contradiction relationships between premises and hypotheses, such as the following:

1. *Premise* Bob is in his room, but because of the thunder and lightning outside, he cannot sleep.
2. *Hypothesis* 1 Bob is awake.
3. *Hypothesis* 2 It is sunny outside.

While the first hypothesis follows from the premise, indicated by the alignment of `cannot sleep` and `awake`, the second hypothesis contradicts the premise, as can be seen from the alignment of `sunny` and `thunder and lightning` and recognizing their incompatibility.

## Code

### ``` train.py```

Accepts the following paramaters as arguments (or none as DEFAULTs for each is set already):
	
 1. ***model_name*** : (string) 'base / lstm / bilastm / bilstm_pool' (DEFAULT set to 'bilstm')
 2. ***nli_path*** : (str) path for NLI data (raw data)
 3. ***glove_path*** : (str) 'path for Glove embeddings (850B, 300D)'
 4. ***lr*** : (int) 'Learning rate for training'
 5. ***checkpoint_path*** : (str) 'Directory to save model during training'
 6. ***outputmodelname*** : (str) 'Name of the saved model'	
 7. ***bsize*** : (int) 'Batch size for training'
 8. ***emb_dim*** : (int) 'Embedding size of word-vectors used'
 9. ***lstm_dim*** : (int) 'Dimension of hidden unit of LSTM'
 10. ***fc_dim*** : (int) 'Dimension of FC Layer (classifier)'
 11. ***n_classes*** : (int) 'Number of classes being predicted for the task'

This is the main function where the training happens and all the other modules declared in *models.py* , *data.py* and *dev_test_evals.py* are called. For further details, check the file for comments in each line and step. NOTE: to start the training process, you just need to run this file with the above mentioned paramaters (optional).


### ```data.py```

This is where raw SNLI data is pre-processed and word-vecs are created using the GLOVE embeddings. It involves the following modules:

 1. ***get_nli()***: This reads the NLI data and partitions it into train,dev and test sets with dictionaries for 's1','s2' and labels.
 2. ***build_vocab()***: This creates a mapping (word-vecs) for each word in the SNLI vocabulary to their respective word embedding in GLOVE.
 3. ***get_batch_from_idx()***: This takes batches of data, pads them to equal lengths and returns the word embeddings for word in each sentence.
	

### ``` models.py```

This file includes all the model classes required for training and is structured as follows:

 1. ***LSTM_main()***: Based on the model being trained, initializes the right class's object in itself. It encodes the sentences pairs as ***u*** and ***v*** by the chosen encoder (LSTM/biLSTM/biLSTM(max-pool) and returns the feature for the classifier in the following form:

> ```concatenate(u, v, |u-v|, u*v)```

 2. ***LSTM()***: Encodes the provided sentences usnig a uni-directional LSTM network and returning the final state as the sentence representation.

 3. ***biLSTM()***: Has 2 options and chooses to implement the right one based on model chosen:

  - *pool = 0* : corresponds to normal biLSTM model that encodes the provided sentences usnig a bi-directional LSTM network and returns the concatenation of final states of both the directions as the sentence representation.
  
  - *pool = 1*: corresponds to the biLSTM(with max-pool) model that takes the concatenation of each hidden state and returns a fixed length vector by taking the maximum over each embedding dimension of the hidden state outputs of the sentences of the batch.
  
### ```dev_test_evals.py```

This file includes module to evaluate the performance of the model after each EPOCH on the *validation set* and at termination on the *test set*. It also saves the model after each EPOCH in the directory provided. 

NOTE: this does not perform *SentEval* evaluation. That is done by the *senteval.py* file.

### ```senteval.py```

Accepts the following paramaters as arguments (or none as DEFAULTs for each is set already):
	
 1. ***model_name*** : (string) 'base / lstm / bilastm / bilstm_pool' (DEFAULT set to 'bilstm')
 2. ***senteval_path*** : (str) path to the cloned 'senteval' directory
 3. ***vec_path*** : (str) 'path for Glove embeddings (850B, 300D)'
 4. ***data_path*** : (str) 'path to the 'data' fodler in senteval directory'
 5. ***model_path*** : (str) 'path to the saved models (pickle files saved during training)'
 
This performs the SentEval evaluation on the following transfer tasks:

```python

transfer_tasks = ['STS14','MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'ImageCaptionRetrieval']
```

and using the following settings:

```python

params_senteval = {'task_path': FLAGS.data_path, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}
```
				 
### ```infer.py``` and ```Demo.ipynb```

They are exactly the same files but in different interfaces. *infer.py* accepts the following arguments:
	
 1. ***model_name*** : (string) 'base / lstm / bilastm / bilstm_pool' (DEFAULT set to 'bilstm')
 2. ***snli_path*** : (str) path to SNLI data to read and pre-process
 3. ***vec_path*** : (str) 'path to Glove embeddings (850B, 300D)'
 4. ***model_path*** : (str) 'path to the saved models (pickle files saved during training)'
 5. ***prem***: (str) The premise sentence as input
 6. ***hyp***: (str) The hypothesis sentence as input

This returns model output (as the weight outputs of 3 classes at the final classifier layer) to give an idea of relative weight given to each choice along with the model's prediction as *entailment, neutral, contradiction*.






	
