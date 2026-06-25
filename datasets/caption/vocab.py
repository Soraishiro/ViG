from __future__ import unicode_literals
from collections import defaultdict
import logging
import os
import json
logger = logging.getLogger(__name__)

class Vocab(object):

    def __init__(self, counter=None, max_size=None, min_freq=1, specials=['<pad>'], vocab_path=None):
        if vocab_path is not None and os.path.exists(vocab_path):
            vocab_data = json.load(open(vocab_path))
            self.freqs = vocab_data['freqs']
            self.itos = vocab_data['itos']
            self.stoi = defaultdict(_default_unk_index)
            self.stoi.update({tok: i for i, tok in enumerate(self.itos)})
        else:
            self.freqs = counter
            if counter is None:
                raise ValueError('Vocab: counter=None and vocab_path not found or invalid')
            counter = counter.copy()
            min_freq = max(min_freq, 1)
            self.itos = list(specials)
            for tok in specials:
                del counter[tok]
            max_size = None if max_size is None else max_size + len(self.itos)
            words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
            words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
            for word, freq in words_and_frequencies:
                if freq < min_freq or len(self.itos) == max_size:
                    break
                self.itos.append(word)
            self.stoi = defaultdict(_default_unk_index)
            self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def extend(self, v, sort=False):
        if not isinstance(v, list):
            words = sorted(v.itos) if sort else v.itos
        else:
            words = set(v)
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1

def _default_unk_index():
    return 0
