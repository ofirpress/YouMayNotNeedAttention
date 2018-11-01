import os
import torch
import pickle

from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

        self.eos_token = "<eos>"
        self.epsilon_token = "@@@"
        self.epsilon_src_token = "@@@@"
        self.start_pad_token = '@str@@'

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):

        if os.path.exists(os.path.join(path, 'dictionary.p')):
            print("Loading existing data!")

            self.dictionary = pickle.load( open(os.path.join(path, 'dictionary.p'), "rb"))

            self.train_src = pickle.load( open(os.path.join(path, 'train_src.p'), "rb"))
            self.valid_src = pickle.load( open(os.path.join(path, 'valid_src.p'), "rb"))

            self.train_trg = pickle.load( open(os.path.join(path, 'train_trg.p'), "rb"))
            self.valid_trg = pickle.load( open(os.path.join(path, 'valid_trg.p'), "rb"))

        else:
            print("Creating pickles of the data! This could take a while...")
            self.dictionary = Dictionary()

            print("Tokenizing train src...")
            self.train_src = self.tokenize(os.path.join(path, 'train_src.txt'))
            print("Tokenizing valid src...")
            self.valid_src = self.tokenize(os.path.join(path, 'valid_src.txt'))
            print("Tokenizing train trg...")
            self.train_trg = self.tokenize(os.path.join(path, 'train_trg.txt'))
            print("Tokenizing valid trg...")
            self.valid_trg = self.tokenize(os.path.join(path, 'valid_trg.txt'))

            pickle.dump(self.train_src, open(os.path.join(path, 'train_src.p'), "wb"), protocol=4)
            pickle.dump(self.valid_src, open(os.path.join(path, 'valid_src.p'), "wb"), protocol=4)

            pickle.dump(self.train_trg, open(os.path.join(path, 'train_trg.p'), "wb"), protocol=4)
            pickle.dump(self.valid_trg, open(os.path.join(path, 'valid_trg.p'), "wb"), protocol=4)

            pickle.dump(self.dictionary, open(os.path.join(path, 'dictionary.p'), "wb"), protocol=4)
            print("Done pickling.")

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path), "The given --data path does not exist."
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + [self.dictionary.eos_token]
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        print("Current vocabulary length: " + str(len(self.dictionary.idx2word)))

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + [self.dictionary.eos_token]
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
