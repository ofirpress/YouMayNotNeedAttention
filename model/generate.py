import argparse

import torch

import math
import pickle
import os
import numpy as np
import time
from sys import stderr

from generate_utils import Sequence, SeqSet

import random
from mosestokenizer import *
import sacrebleu

parser = argparse.ArgumentParser(description='Translate with a trained model. Optionally, this script can also calculate BLEU.')
parser.add_argument('--data', type=str, default=' ~/corpus/WMTENDE/5pad/',
                    help='location of the data corpus')

parser.add_argument('--save_dir', type=str, default='./output/',
                    help='Folder in which to save the generated translation (file name will be generated automatically)')

parser.add_argument('--src_path', type=str, default='./data/valid_src.txt',
                    help='location of the file to translate')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--beam_size', type=int, default=10,
                    help='size of beam')

parser.add_argument('--start_pads', type=int, default=3,
                    help='number of starting pad symbols in each sentence. Set to 0 for no start padding.')
parser.add_argument('--epsilon_limit', type=int, default=9,
                    help='limit on number of epsilons')
parser.add_argument('--src_epsilon_injection', type=int, default=3,
                    help='number of source epsilon tokens to inject into the source sentence before the <eos> tag')
parser.add_argument('--debug', action='store_true',
                    help='debug mode')

parser.add_argument('--eval', action='store_true',
                    help='compute BLEU after generating translations')
parser.add_argument('--target_translation',
                    help='The correct translation of the source file into the target language')

parser.add_argument('--language', type=str, default='en',
                    help='Target language. ')

parser.add_argument('--id', help='optional. useful when running this script multiple times in parallel.')

args = parser.parse_args()

print('Args: {}'.format(args), file=stderr)

if(args.eval and args.debug):
    print ("Cant eval and debug in same run. Please disable at least one of these options.")
    from sys import exit
    exit()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

dictionary = pickle.load(open(os.path.join(args.data, 'dictionary.p'), "rb"))
ntokens = len(dictionary)

bos = dictionary.word2idx[dictionary.eos_token]
eos = dictionary.word2idx[dictionary.eos_token]
epsilon = dictionary.word2idx[dictionary.epsilon_token]
epsilon_src = dictionary.word2idx[dictionary.epsilon_src_token]


if args.start_pads > 0:
    start_pad = dictionary.word2idx[dictionary.start_pad_token]
special_tokens = [dictionary.epsilon_token] + [dictionary.epsilon_src_token] + [dictionary.eos_token] + ([dictionary.start_pad_token] if args.start_pads > 0 else [])

MAX_TRG_FURTHER = 10
start_seq = [bos]


def clean_sentence(sent, special_tokens): #make sure the input sentence does not have any special tokens.
    cleaned_sent = []
    for word in sent:
        if word not in special_tokens:
            cleaned_sent.append(word)
    return cleaned_sent


beam_size = args.beam_size

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

log_soft_max = torch.nn.LogSoftmax(dim=-1)

start_time = time.time()

default_inital_state = model.init_hidden(1)
output_sentences = []
with open(args.src_path, 'r') as f:
    for line_number, line in enumerate(f):
        with torch.no_grad():

            #print (line_number)

            src_eos_reached = False
            src_eos_index = -1 # -1 is just a placeholder

            sent = clean_sentence(line.split(), special_tokens) + ['<eos>']

            if args.debug:
                print(">"+ " ".join(sent))

            beam_top = SeqSet(beam_size)

            initial_line_state = default_inital_state


            for i in range(beam_size):
                beam_top.append(Sequence(sentence=start_seq, logprob=0.0, state=initial_line_state, last_token=bos))


            model_time = 0
            update_time = 0

            EOSed_sequences = []
            for i in range(1000): # so 1000 is the definite maximal output length, but in practice we don't get even close to that
                if src_eos_reached and i - src_eos_index > args.src_epsilon_injection:
                    break # trg sentence length will not be more than (index at which src emitted <eos>) + MAX_TRG_FURTHUR

                current_best = beam_top.extract()
                current_beam_size = len(current_best)  # could be smaller than beam_size because of pruned sentences that reached EOS token

                if current_beam_size == 0:
                    break

                input = -1*torch.ones((1, current_beam_size)).long() # -1 is placeholder
                if args.cuda:
                    input.data = input.data.cuda()

                prev_tokens = [seq.last_token for seq in current_best]
                prev_target = torch.Tensor([prev_tokens]).long()
                if args.cuda:
                    prev_target.data = prev_target.data.cuda()

                states = [seq.state for seq in current_best]
                nlayers = len(states[0])
                prev_state = [(torch.cat([state[layer][0] for state in states], dim=1), torch.cat([state[layer][1] for state in states], dim=1)) for layer in range(nlayers)]

                try:
                     input_token = epsilon_src if i >= len(sent) else dictionary.word2idx[sent[i]]
                except KeyError:
                    print('Unkown token: {}'.format(sent[i]), file=stderr)
                    input_token = epsilon_src #shouldn't really ever get here if you use BPE


                if input_token == eos:
                    src_eos_reached = True
                    src_eos_index = i

                if args.src_epsilon_injection > 0 and (input_token == eos or input_token == epsilon_src): #this controls the epsilon injection
                    input_tokens = [epsilon_src] + [eos]
                else:
                    input_tokens = [input_token]

                for curr_input_token in input_tokens:
                    input.data = input.data.fill_(curr_input_token)

                    start2_time = time.time()
                    output, hidden = model(input, prev_target, prev_state)
                    model_time += time.time() - start2_time

                    word_weights = output.squeeze()

                    log_soft_maxed = log_soft_max(word_weights).data.cpu()

                    if current_beam_size == 1:
                        log_soft_maxed = np.expand_dims(log_soft_maxed, axis=0)

                    for be in range(current_beam_size):
                        if not src_eos_reached or curr_input_token == epsilon_src: # model shouldn't emit eos if src hasn't finished inputting the sentence.
                            log_soft_maxed[be][eos] = -100000

                        if args.start_pads > 0 and i >= args.start_pads:
                            log_soft_maxed[be][start_pad] = -100000

                        if i> args.epsilon_limit and current_best[be].number_epsilons >= args.epsilon_limit:
                            log_soft_maxed[be][epsilon] = -100000

                    log_probs = np.array([seq.logprob for seq in current_best])
                    log_probs = np.expand_dims(log_probs, axis=1)

                    if i == 0 or (args.start_pads > 0 and i == args.start_pads):
                        log_soft_maxed= log_soft_maxed[0]
                        log_soft_maxed = np.expand_dims(log_soft_maxed, axis=0)
                        log_probs = log_probs[0]

                    new_log_probs = log_probs + log_soft_maxed

                    new_log_probs = np.reshape(new_log_probs, (-1))

                    current_top = np.argpartition(new_log_probs, -beam_size)[-beam_size:] #k-argmax where k is beam_size

                    start3_time = time.time()

                    for c_t in current_top:
                        seq_number = int(np.floor(c_t/ntokens))
                        word = int(c_t) - seq_number*ntokens

                        if i < args.start_pads:
                            word = start_pad

                        new_sentence = current_best[seq_number].sentence + [word]

                        if args.start_pads == 0 or word != start_pad:
                            previous_logprob  = current_best[seq_number].logprob
                            current_logprob =  log_soft_maxed[seq_number][word]


                            if(isinstance(current_logprob, torch.Tensor)):
                                current_logprob = current_logprob.item()

                            logprob = previous_logprob + current_logprob
                        else:
                            logprob = 0


                        c_t_hidden = [( hidden[layer][0][:,seq_number,:].unsqueeze(0) , hidden[layer][1][:,seq_number,:].unsqueeze(0) ) for layer in range(nlayers)]
                        score = logprob / (i+1)
                        number_epsilons = current_best[seq_number].number_epsilons + 1 if word == epsilon else current_best[seq_number].number_epsilons

                        if word != eos or i == 0:
                            beam_top.append(Sequence(new_sentence, c_t_hidden, logprob, word, score=score, number_epsilons=number_epsilons))

                        if word == eos and i > 0:
                            EOSed_sequences.append(Sequence(new_sentence, c_t_hidden, logprob, word, score=score, number_epsilons=number_epsilons))

                    del output, hidden, log_soft_maxed

                del input, prev_target, prev_state


            EOSed_sequences.sort(reverse=True)

            if not args.debug:
                if len(EOSed_sequences) > 0:
                    best = EOSed_sequences[0]
                else:
                    best = beam_top.extract(sort=True)[0]

                sentence = [dictionary.idx2word[w] for w in best.sentence]
                sentence = clean_sentence(sentence, special_tokens)

                output_sentences.append(" ".join(sentence))


            if args.debug:
                if len(EOSed_sequences) > 0:
                    for seq in EOSed_sequences:
                        sentence = [dictionary.idx2word[w] for w in seq.sentence]
                        l = len(sentence)
                        print(str(l) + "  " + " ".join(sentence) + " " + str(seq.logprob) + " " + str(seq.logprob/l))

                print('>>>>')
                not_EOSed = beam_top.extract(sort=True)
                if len(not_EOSed) > 0:
                    for seq in not_EOSed:
                        sentence = [dictionary.idx2word[w] for w in seq.sentence]
                        l = len(sentence)
                        print(str(l) + "  " + " ".join(sentence) + " " + str(seq.logprob) + " " + str(seq.logprob/l))

                if len(EOSed_sequences) > 0:
                    best = EOSed_sequences[0]
                else:
                    best = beam_top.extract(sort=True)[0]



if args.debug:
    print(time.time() - start_time)



s='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
save_file_name = ''.join(random.sample(s,10))

save_path = os.path.join(args.save_dir,  save_file_name)
with open(save_path, 'w') as thefile:

    for item in output_sentences:
        item = item.replace("###", " ")


        item = item.replace("@@@ ", "")
        item = item.replace("@@@", "")
        item = item.replace("@@ ", "")

        with MosesDetokenizer(args.language) as detokenize:
            item = detokenize(item.split(" "))


        thefile.write("%s\n" % item)


if args.eval:
    inputfh = open(save_path, 'r')
    system = inputfh.readlines()

    inputref = open(args.target_translation, 'r')
    ref = inputref.readlines()

    print(str(args.id) + "  "+ str(sacrebleu.corpus_bleu(system, [ref]).score) + " " +save_path )



