import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import os, sys

import data

from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint

parser = argparse.ArgumentParser(description='Eager Translation Model')
parser.add_argument('--data', type=str, default='./data/',
                    help='location of the data corpus')

parser.add_argument('--emsize', type=int, default=1150,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=4,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=25,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=120,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to rnn output')
parser.add_argument('--dropouth', type=float, default=0.2,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.3,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.25,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--dropoutcomb', type=float, default=0.1,
                    help='dropout for combined input embeddings')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')


parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--update_interval', type=int, default=6500, metavar='N',
                    help='update interval')
parser.add_argument('--start_decaying_lr_step', type=int, default=45000, metavar='N',
                    help='update interval')


parser.add_argument('--save', type=str, default='exp',
                    help='path to save the folder which will contain the final model and logs')

parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
# parser.add_argument('--continue_train', action='store_true',
#                     help='continue train from a checkpoint')

args = parser.parse_args()

import model


args.save = '{}{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
create_exp_dir(args.save, scripts_to_save=['main.py', 'model.py'])


def logging(s, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(os.path.join(args.save, 'log.txt'), 'a+') as f_log:
            f_log.write(s + '\n')


# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed_all(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1

train_data_src, train_data_trg = batchify(corpus.train_src, corpus.train_trg, args.batch_size, args)
val_data_src, val_data_trg = batchify(corpus.valid_src, corpus.valid_trg, eval_batch_size, args)
test_data_src, test_data_trg = batchify(corpus.valid_src, corpus.valid_trg, test_batch_size,
                                        args)  # test data is same as valid data, we just use different batch size

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)

if args.continue_train:  # probably needs to be fixed
    model = torch.load(os.path.join(args.save, 'model.pt'))
    print("Loaded existing model.")
else:
    model = model.RNNModel('LSTM', ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth,
                           args.dropouti, args.dropoute, args.wdrop, args.dropoutcomb, args.tied)

    weight = torch.ones(len(corpus.dictionary))
    epsilon = corpus.dictionary.word2idx["@@@"]

    criterion = nn.CrossEntropyLoss()

if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
logging('Args: {}'.format(args))
logging('Model total parameters: {}'.format(total_params))


###############################################################################
# Training code
###############################################################################

def evaluate(data_source_src, data_source_trg, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_eval_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source_src.size(0) - 1, args.bptt):
        data, prev_targets, targets = get_batch(data_source_src, data_source_trg, i, args, evaluation=True)
        output, hidden = model(data, prev_targets, hidden)
        output_flat = output.view(-1, ntokens)
        total_eval_loss += len(data) * criterion(output_flat, targets).data


        hidden = repackage_hidden(hidden)
    return total_eval_loss.item() / len(data_source_src)


def train(step_number, stored_loss, lr):
    # Turn on training mode which enables dropout.
    total_loss = 0

    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    i = 0
    while i < train_data_src.size(0) - 1 - 1:
        bptt = args.bptt #if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = bptt#max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        model.train()
        data, prev_targets, targets = get_batch(train_data_src, train_data_trg, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, prev_targets, hidden, return_h=True)
        raw_loss = criterion(output.view(-1, ntokens), targets)

        loss = raw_loss
        # Activiation Regularization
        loss = loss + sum( dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        loss = loss + sum( (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += raw_loss.data

        if step_number % args.log_interval == 0 and step_number > 0:
            cur_loss = total_loss.item() / args.log_interval

            elapsed = time.time() - start_time #timer doesnt stop while validating, so this will be wrong
                                               #if there was a validation call since the last log print
            logging('| epoch {:3d} | step {:5d} | {:5d} steps per epoch | lr {:01.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}  |input tkn/s {:7.2}'.format(
                epoch, step_number, len(train_data_src) // args.bptt, lr,
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss),
                              args.log_interval * args.bptt * args.batch_size / elapsed))

            total_loss = 0
            start_time = time.time()

        ###


        if step_number % args.update_interval == 0 and step_number > 0:
            val_loss = evaluate(val_data_src, val_data_trg, eval_batch_size)
            logging('|VALIDATION| step number {:3d} | valid loss {:5.2f} | '
                    'valid ppl {:8.2f} |  '.format(step_number,
                                                   val_loss, math.exp(val_loss)))

            save_checkpoint(model, optimizer, args.save, suffix=str(step_number))  # just for debug


            # save_checkpoint(model, optimizer, args.save, suffix="last")

            if step_number > args.start_decaying_lr_step:

                if math.exp(val_loss) < math.exp(stored_loss):  
                    # save_checkpoint(model, optimizer, args.save)
                    # logging('Saving Normal!')
                    stored_loss = val_loss

                else:
                    lr *= 0.5
                    print('Lowering LR to: ' + str(lr))
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr





        step_number += 1
        i += seq_len

        del data, targets, loss, raw_loss
        del output, rnn_hs, dropped_rnn_hs
    return step_number, stored_loss, lr


# Loop over epochs.
lr = args.lr
stored_loss = 20
step_number =0
# At any point you can hit Ctrl + C to break out of training early.
try:
    # if args.continue_train:
    # optimizer_state = torch.load(os.path.join(args.save, 'optimizer.pt'))
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    # optimizer.load_state_dict(optimizer_state)
    # else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        step_number, stored_loss, lr = train(step_number, stored_loss, lr)

        val_loss = evaluate(val_data_src, val_data_trg, eval_batch_size)
        logging('-' * 89)
        logging('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} |  '.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
        logging('-' * 89)



except KeyboardInterrupt:
    logging('-' * 89)
    logging('Exiting from training early')

