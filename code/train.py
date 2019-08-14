from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse, time, datetime
import pprint
import os
import sys
import time
import argparse

import numpy as np

import torch
import torchtext
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from nltk import word_tokenize
import nltk
# nltk.download('punkt')
from torch.autograd import Variable
from torchtext.data import Field, BucketIterator
from torchtext import datasets
import torch.nn as nn

from data import get_nli, build_vocab, get_batch_from_idx, SNLIBatchGenerator
from models import SNLI
from dev_test_evals import model_eval



def prepare_training():
    # Define fields for reading SNLI data
    start = time.time()
    print("="*80 + "\n\t\t\t\tPreparing Data\n" + "="*80 + "\n")
    print("\n==>> Preparing train-dev-test splits of SNLI data...\n")
    TEXT = Field(sequential=True, tokenize=word_tokenize, lower=True, use_vocab=True, batch_first=False, include_lengths=True)
    LABEL = Field(sequential=False, use_vocab=True, pad_token=None, unk_token=None, batch_first=False)
    train, dev, test = datasets.SNLI.splits(TEXT, LABEL, root = config['nli_path'])

    print("\n==>> Loading GloVe embeddings...\n")
    glove_embeds = torchtext.vocab.Vectors(name= config['glove_path'], max_vectors= config['embed_dim'])

    print("\n==>> Building vocabulary...")
    TEXT.build_vocab(train, dev, vectors = glove_embeds)
    LABEL.build_vocab(train)
    vocab_size = len(TEXT.vocab)
    config['vocab_size'] = vocab_size
    print("Vocabulary size = ", vocab_size)

    # Set 'unk' embeddings as the mean of all the embeddings
    TEXT.vocab.vectors[TEXT.vocab.stoi['<unk>']] = torch.mean(TEXT.vocab.vectors, dim=0)

    # Define the iterator over the train and valid set
    train_iter, dev_iter, test_iter = BucketIterator.splits(datasets=(train, dev, test),
                                                   batch_sizes=(config['batch_size'], config['batch_size'], config['batch_size']),
                                                   sort_key=lambda x: x.premise,
                                                   shuffle=True,
                                                   sort_within_batch=True,
                                                   device=device)
    # Custom wrapper over the iterators
    train_batch_loader = SNLIBatchGenerator(train_iter)
    dev_batch_loader = SNLIBatchGenerator(dev_iter)
    test_batch_loader = SNLIBatchGenerator(test_iter)

    end = time.time()
    hours, minutes, seconds = calc_elapsed_time(start, end)

    print("\n"+ "-"*50 + "\nTook  {:0>2}:{:0>2}:{:05.2f}  to Prepare Data\n".format(hours,minutes,seconds))

    return train_batch_loader, dev_batch_loader, test_batch_loader, TEXT, LABEL


def calc_elapsed_time(start, end):
    hours, rem = divmod(end-start, 3600)
    time_hours, time_rem = divmod(end, 3600)
    minutes, seconds = divmod(rem, 60)
    time_mins, _ = divmod(time_rem, 60)
    return int(hours), int(minutes), seconds


def eval_network(model, test = False):
    eval_acc = 0
    batch_loader = test_batch_loader if test else dev_batch_loader
    model.eval()
    with torch.no_grad():
        for iters, (premise, hyp, label) in enumerate(batch_loader):
            out = model(premise[0], premise[1], hyp[0], hyp[1])
            preds = torch.argmax(out, dim=1)
            accuracy = torch.sum(preds == label, dtype=torch.float32) / out.shape[0]
            eval_acc += accuracy
        eval_acc /= iters
    return eval_acc



def print_stats(epoch, train_loss, train_acc, val_acc, start):
    end = time.time()
    hours, minutes, seconds = calc_elapsed_time(start, end)
    print(("Epoch: {}/{},    train_loss: {:.4f},  train_acc = {:.2f}   eval_acc = {:.2f}  | Elapsed Time:  {:0>2}:{:0>2}:{:05.2f}"
                     .format(epoch, config['max_epoch'], train_loss, train_acc, val_acc, hours,minutes,seconds)))



def train_network():
    print("\n" + "="*80 + "\n\t\t\t\tTraining Network\n" + "="*80 + "\n")

    # Seeds for reproduceable runs
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize the model, optimizer and loss function
    model = SNLI(config, pretrained_vectors = TEXT.vocab.vectors).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'], momentum = config['momentum'])
    criterion = nn.CrossEntropyLoss()

    # Load the checkpoint to resume training if found
    model_file = os.path.join(config['checkpoint_path'], config['outputmodelname'])
    if os.path.isfile(model_file):
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print("\nResuming training from epoch %d with loaded model and optimizer..." % start_epoch)
    else:
        start_epoch = 1
        print("\nNo Checkpoints found for the chosen model to reusme training... \nTraining the  ''{}''  model from scratch...".format(config['model_name']))

    start = time.time()
    best_val_acc = 0
    prev_val_acc = 0
    total_iters = 0
    terminate_training = False
    print("\nStarting time of training:  {} \n".format(datetime.datetime.now()))
    for epoch in range(start_epoch, config['max_epoch']+1):
        train_loss = 0
        train_acc = 0
        # TP, FP, TN, FN = 0, 0, 0, 0
        model.train()
        for iters, (premise, hyp, label) in enumerate(train_batch_loader):
            out = model(premise[0], premise[1], hyp[0], hyp[1])
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(out, dim=1)
            accuracy = torch.sum(preds == label, dtype=torch.float32) / out.shape[0]
            train_loss += loss.detach().item()
            train_acc += accuracy
            if iters%500 == 0:
                writer.add_scalar('Train/iters/loss', train_loss/(iters+1), ((iters+1)+ total_iters))
                writer.add_scalar('Train/iters/accuracy', train_acc/(iters+1)*100, ((iters+1)+ total_iters))
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    writer.add_histogram('iters/'+name, param.data.view(-1), global_step= ((iters+1)+total_iters))

        total_iters += iters
        train_loss = train_loss/iters
        train_acc = (train_acc/iters)*100

        # Evaluate on test set
        val_acc = eval_network(model)*100

        # print stats
        print_stats(epoch, train_loss, train_acc, val_acc, start)

        # write stats to tensorboard
        # TP += ((preds == label).float() * (preds == 1).float()).sum(dim=(0,1)).cpu().data.numpy()
        # FP += ((preds != label).float() * (preds == 1).float()).sum(dim=(0,1)).cpu().data.numpy()
        # TN += ((preds == label).float() * (preds == 0).float()).sum(dim=(0,1)).cpu().data.numpy()
        # FN += ((preds != label).float() * (preds == 0).float()).sum(dim=(0,1)).cpu().data.numpy()

        # precision_per_label = TP / (TP + FP + 1e-10)
        # recall_per_label = TP / (TP + FN + 1e-10)
        # f1_per_label = 2 * precision_per_label * recall_per_label / (1e-5 + precision_per_label + recall_per_label)

        writer.add_scalar('Train/epochs/loss', train_loss, epoch+1)
        writer.add_scalar('Train/epochs/accuracy', train_acc, epoch+1)
        writer.add_scalar('Validation/acc', val_acc, epoch+1)
        # writer.add_scalar("precision", precision_per_label, epoch)
        # writer.add_scalar("recall", recall_per_label, epoch)
        # writer.add_scalar("f1", f1_per_label, epoch)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            writer.add_histogram('epochs/' + name, param.data.view(-1), global_step= epoch+1)

        # Save model checkpoints for best model
        if val_acc > best_val_acc:
            print("\nNew High Score! Saving model...")
            best_val_acc = val_acc
            # Save the state and the vocabulary
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'text_vocab': TEXT.vocab.stoi,
                'label_vocab': LABEL.vocab.stoi
            }, os.path.join(config['checkpoint_path'], config['outputmodelname']))

        # If validation accuracy does not improve, divide the learning rate by 5 and
        # if learning rate falls below 1e-5 terminate training
        if val_acc <= prev_val_acc:
            for param_group in optimizer.param_groups:
                if param_group['lr'] < 1e-5:
                    terminate_training = True
                    break
                param_group['lr'] /= 5
                print("Learning rate changed to :  ", param_group['lr'])

        prev_val_acc = val_acc
        if terminate_training:
            break

    # Termination message
    if terminate_training:
        print("\n" + "-"*100 + "\nTraining terminated because the learning rate fell below:  %f" % 1e-5)
    else:
        print("\n" + "-"*100 + "\nMaximum epochs reached. Finished training !!")

    print("\n" + "-"*50 + "\n\t\t\tEvaluating on test set\n" + "-"*50)
    model_file = os.path.join(config['checkpoint_path'], config['outputmodelname'])
    if os.path.isfile(model_file):
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError("\nNo Saved model state_dict found for the chosen model...!!! \nAborting evaluation on test set...".format(config['model_name']))
    test_acc = eval_network(model, test = True)*100
    print("\n" + "="*50 + "Test accuracy of best model = {:.2f}%".format(test_acc))

    writer.close()
    return None




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required Paths
    parser.add_argument('--nli_path', type = str, default = './data',
                          help='path for NLI data (raw data)')
    parser.add_argument('--glove_path', type = str, default = './data/glove/glove.840B.300d.txt',
                          help='path for Glove embeddings (850B, 300D)')
    parser.add_argument('--checkpoint_path', type = str, default = './checkpoints',
                          help='Directory of check point')
    parser.add_argument("--outputmodelname", type=str, default= 'best_model.pt',
                       help = 'saved model name')

    # Training Params
    parser.add_argument('--model_name', type = str, default = 'base',
                          help='model name: base / lstm / bilstm / lstm')
    parser.add_argument('--lr', type = float, default = 0.1,
                          help='Learning rate for training')
    parser.add_argument('--batch_size', type = int, default = 32,
                          help='batch size for training"')
    parser.add_argument('--embed_dim', type = int, default = 300,
                          help='dimen of word embeddings used"')
    parser.add_argument('--lstm_dim', type = int, default = 2048,
                          help='dimen of hidden unit of LSTM"')
    parser.add_argument('--fc_dim', type = int, default = 512,
                          help='dimen of FC layer"')
    parser.add_argument('--n_classes', type = int, default = 3,
                          help='number of classes"')
    parser.add_argument('--optimizer', type = str, default = 'SGD',
                        help = 'Optimizer to use for training')
    parser.add_argument('--dpout', type = float, default = 0.1,
                        help = 'Dropout for training')
    parser.add_argument('--weight_decay', type = float, default = 1e-4,
                        help = 'weight decay for optimizer')
    parser.add_argument('--momentum', type = float, default = 0.8,
                        help = 'Momentum for optimizer')
    parser.add_argument('--max_epoch', type = int, default = 50,
                        help = 'Max epochs to train for')

    args, unparsed = parser.parse_known_args()
    config = args.__dict__

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    config['device'] = device

    global dtype
    dtype = torch.FloatTensor

    # Check all provided paths:
    if not os.path.exists(config['nli_path']):
        raise ValueError("[!] ERROR: NLI data path does not exist")
    if not os.path.exists(config['glove_path']):
        raise ValueError("[!] ERROR: Glove Embeddings path does not exist")
    if not os.path.exists(config['checkpoint_path']):
        print("\nCreating checkpoint path for output videos:  ", config['checkpoint_path'])
        os.makedirs(config['checkpoint_path'])
    if config['model_name'] not in ['base', 'lstm', 'bilstm' , 'bilstm_pool']:
        raise ValueError("{!} ERROR:  model_name is incorrect. Choose one of base/lstm/bilstm/bilstm_pool")


    # Prepare the tensorboard writer
    writer = SummaryWriter(os.path.join('logs', config['model_name']))

    # Prepare the datasets and iterator for training and evaluation
    train_batch_loader, dev_batch_loader, test_batch_loader, TEXT, LABEL = prepare_training()

    #Print args
    print("\n" + "x"*50 + "\n\nRunning training with the following parameters: \n")
    for key, value in config.items():
        print(key + ' : ' + str(value))
    print("\n" + "x"*50)
    train_network()

