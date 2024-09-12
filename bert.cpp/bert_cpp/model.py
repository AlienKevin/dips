import os
import sys
import ctypes
import numpy as np
from tqdm import tqdm

from . import api
from .utils import suppress_stdout_stderr

# increment ctypes pointer
def increment_pointer(p, d):
    t = type(p)._type_
    v = ctypes.cast(p, ctypes.c_void_p)
    v.value += d * ctypes.sizeof(t)
    return ctypes.cast(v, ctypes.POINTER(t))

# main bert interface
class BertModel:
    def __init__(self, fname, max_tokens=None, batch_size=32, use_cpu=False, allocate=True, verbose=False):
        # load model from file
        with suppress_stdout_stderr(disable=verbose):
            self.ctx = api.bert_load_from_file(fname, use_cpu)
        if not self.ctx:
            raise ValueError(f'Failed to load model from file: {fname}')

        # get model dimensions
        self.verbose = verbose
        self.n_embd = api.bert_n_embd(self.ctx)
        self.n_max_tokens = api.bert_n_max_tokens(self.ctx)

        # allocate compute buffers
        if allocate:
            self.allocate(batch_size, max_tokens)

    def __del__(self):
        api.bert_free(self.ctx)

    def allocate(self, batch_size, max_tokens=None):
        self.batch_size = batch_size
        self.max_tokens = max_tokens if max_tokens is not None else self.n_max_tokens
        with suppress_stdout_stderr(disable=self.verbose):
            api.bert_allocate_buffers(self.ctx, self.max_tokens, self.batch_size)

    def tokenize(self, text, max_tokens=None):
        max_tokens = self.n_max_tokens if max_tokens is None else max_tokens
        return api.bert_tokenize(self.ctx, text, max_tokens)

    def detokenize(self, tokens, max_fact=16, debug=False):
        max_len = len(tokens) * max_fact
        return api.bert_detokenize(self.ctx, tokens, max_len, debug)

    def embed_batch(self, batch, output=None, normalize=True, n_threads=8):
        if output is None:
            return api.bert_encode_batch(self.ctx, batch, normalize, n_threads)
        else:
            api.bert_encode_batch_c(self.ctx, batch, output, normalize, n_threads)

    def embed(self, text, progress=False, **kwargs):
        # handle singleton case
        if isinstance(text, str):
            text = [text]
            squeeze = True
        else:
            squeeze = False
        n_input = len(text)

        # create embedding memory
        embed = np.zeros((n_input, self.n_embd), dtype=np.float32)
        embed_p = embed.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # loop over batches
        indices = range(0, n_input, self.batch_size)
        if progress:
            indices = tqdm(list(indices))
        for i in indices:
            j = min(i + self.batch_size, n_input)
            batch = text[i:j]
            batch_p = increment_pointer(embed_p, i * self.n_embd)
            self.embed_batch(batch, output=batch_p, **kwargs)

        # return squeezed maybe
        return embed[0] if squeeze else embed
