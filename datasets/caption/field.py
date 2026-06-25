import json
import six
import h5py
import torch
import numpy as np
from PIL import Image
from itertools import chain
from collections import Counter, OrderedDict
from tqdm import tqdm
from .vocab import Vocab
from .utils import get_tokenizer
import os
try:
    import spacy
except Exception:
    spacy = None

def _build_spacy_tokenizer():
    if spacy is None:
        return None
    for model_name in ('en_core_web_sm', 'en'):
        try:
            return spacy.load(model_name).tokenizer
        except Exception:
            continue
    try:
        return spacy.blank('en').tokenizer
    except Exception:
        return None
SPACY_TOKENIZER = _build_spacy_tokenizer()

class ImageField:

    def __init__(self, hdf5_path=None, transform=None, use_reg_feat=False, use_gri_feat=False, use_hdf5_feat=False, **kwargs):
        self.hdf5_path = hdf5_path
        self.use_hdf5_feat = use_hdf5_feat
        self.use_reg_feat = use_reg_feat
        self.use_gri_feat = use_gri_feat
        self.transform = transform

    def init_hdf5_feat(self):
        self.use_hdf5_feat = True
        with h5py.File(self.hdf5_path, 'r') as f:
            self.image_ids = f['image_ids'][:len(f['image_ids'])]
            self.img_id2idx = {img_id: img_idx for img_idx, img_id in enumerate(self.image_ids)}

    def preprocess(self, path, image_id=None):
        if self.use_hdf5_feat:
            if image_id is None:
                image_id = int(path.split('_')[-1].split('.')[0])
            outputs = {}
            img_idx = self.img_id2idx[image_id]
            if self.use_gri_feat:
                with h5py.File(self.hdf5_path, 'r') as h:
                    outputs['gri_feat'] = torch.from_numpy(h['gri_feat'][img_idx])
                    outputs['gri_mask'] = torch.from_numpy(h['gri_mask'][img_idx])
            if self.use_reg_feat:
                with h5py.File(self.hdf5_path, 'r') as h:
                    outputs['reg_feat'] = torch.from_numpy(h['reg_feat'][img_idx])
                    outputs['reg_mask'] = torch.from_numpy(h['reg_mask'][img_idx])
            return outputs
        else:
            img = Image.open(path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        return img

def tokenize(s):
    if SPACY_TOKENIZER is None:
        return str(s).split()
    return [tok.text for tok in SPACY_TOKENIZER(s)]

class TextField:
    vocab_cls = Vocab
    dtypes = {torch.float32: float, torch.float: float, torch.float64: float, torch.double: float, torch.float16: float, torch.half: float, torch.uint8: int, torch.int8: int, torch.int16: int, torch.short: int, torch.int32: int, torch.int: int, torch.int64: int, torch.long: int}
    punctuations = ["''", "'", '``', '`', '-LRB-', '-RRB-', '-LCB-', '-RCB-', '.', '?', '!', ',', ':', '-', '--', '...', ';']

    def __init__(self, use_vocab=True, init_token='<bos>', eos_token='<eos>', fix_length=None, dtype=torch.long, lower=True, tokenize='spacy', remove_punctuation=True, include_lengths=False, batch_first=True, pad_token='<pad>', unk_token='<unk>', pad_first=False, truncate_first=False, vectors=None, nopoints=False, vocab_path='', build_vocab=False, **kwargs):
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.lower = lower
        self.remove_punctuation = remove_punctuation
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_first = pad_first
        self.truncate_first = truncate_first
        self.vocab_path = vocab_path
        if build_vocab:
            self.vocab = None
        else:
            self.vocab = Vocab(vocab_path=vocab_path)
        self.vectors = vectors
        if nopoints:
            self.punctuations.append('..')

    def preprocess(self, caption):
        if six.PY2 and isinstance(caption, six.string_types) and (not isinstance(caption, six.text_type)):
            caption = six.text_type(caption, encoding='utf-8')
        if self.lower:
            caption = six.text_type.lower(caption)
        caption = tokenize(caption.rstrip('\n'))
        if self.remove_punctuation:
            caption = [w for w in caption if w not in self.punctuations]
        return caption

    def process(self, batch, device=None):
        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device)
        return tensor

    def build_vocab(self, *sources, **kwargs):
        counter = Counter()
        for data in sources:
            for x in tqdm(data):
                x = self.preprocess(x)
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))
        specials = list(OrderedDict.fromkeys([tok for tok in [self.unk_token, self.pad_token, self.init_token, self.eos_token] if tok is not None]))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)
        with open(self.vocab_path, 'w') as f:
            json.dump({'itos': self.vocab.itos, 'freqs': self.vocab.freqs}, f)

    def pad(self, minibatch):
        minibatch = list(minibatch)
        if self.fix_length is None:
            max_len = max((len(x) for x in minibatch))
        else:
            max_len = self.fix_length + (self.init_token, self.eos_token).count(None) - 2
        padded, lengths = ([], [])
        for x in minibatch:
            if self.pad_first:
                padded.append([self.pad_token] * max(0, max_len - len(x)) + ([] if self.init_token is None else [self.init_token]) + list(x[-max_len:] if self.truncate_first else x[:max_len]) + ([] if self.eos_token is None else [self.eos_token]))
            else:
                padded.append(([] if self.init_token is None else [self.init_token]) + list(x[-max_len:] if self.truncate_first else x[:max_len]) + ([] if self.eos_token is None else [self.eos_token]) + [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return (padded, lengths)
        return padded

    def numericalize(self, arr, device=None):
        if self.include_lengths and (not isinstance(arr, tuple)):
            raise ValueError('Field has include_lengths set to True, but input data is not a tuple of (data batch, batch lengths).')
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)
        if self.use_vocab:
            arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]
            var = torch.tensor(arr, dtype=self.dtype, device=device)
        else:
            if self.vectors:
                arr = [[self.vectors[x] for x in ex] for ex in arr]
            numericalization_func = self.dtypes[self.dtype]
            arr = [numericalization_func(x) if isinstance(x, six.string_types) else x for x in arr]
            var = torch.cat([torch.cat([a.unsqueeze(0) for a in ar]).unsqueeze(0) for ar in arr])
        if not self.batch_first:
            var.t_()
        var = var.contiguous()
        if self.include_lengths:
            return (var, lengths)
        return var

    def decode(self, word_idxs, join_words=True):
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            return self.decode([word_idxs], join_words)[0]
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self.decode([word_idxs], join_words)[0]
        elif isinstance(word_idxs, np.ndarray) and word_idxs.ndim == 1:
            return self.decode(word_idxs.reshape((1, -1)), join_words)[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self.decode(word_idxs.unsqueeze(0), join_words)[0]
        captions = []
        for wis in word_idxs:
            caption = []
            for wi in wis:
                word = self.vocab.itos[int(wi)]
                if word == self.eos_token:
                    break
                caption.append(word)
            if join_words:
                caption = ' '.join(caption)
            captions.append(caption)
        return captions
