# low level bert.cpp bindings

import ctypes
import numpy as np

from .utils import load_shared_library

# load the library
_lib = load_shared_library('bert')

#
# set up ctypes for library
#

# load model from file
_lib.bert_load_from_file.restype = ctypes.c_void_p
_lib.bert_load_from_file.argtypes = [
    ctypes.c_char_p, # const char * fname
    ctypes.c_bool,   # bool use_cpu
]
def bert_load_from_file(fname, use_cpu=False):
    return _lib.bert_load_from_file(fname.encode('utf-8'), use_cpu)

# allocate compute buffers
_lib.bert_allocate_buffers.restype = ctypes.c_void_p
_lib.bert_allocate_buffers.argtypes = [
    ctypes.c_void_p, # bert_ctx * ctx
    ctypes.c_int32,  # int32_t n_max_tokens
    ctypes.c_int32,  # int32_t batch_size
]
def bert_allocate_buffers(ctx, n_max_tokens, batch_size):
    return _lib.bert_allocate_buffers(ctx, n_max_tokens, batch_size)

# get model embed dimensions
_lib.bert_n_embd.restype = ctypes.c_int32
_lib.bert_n_embd.argtypes = [
    ctypes.c_void_p # struct bert_ctx * ctx
]
def bert_n_embd(ctx):
    return _lib.bert_n_embd(ctx)

# get model max tokens
_lib.bert_n_max_tokens.restype = ctypes.c_int32
_lib.bert_n_max_tokens.argtypes = [
    ctypes.c_void_p # struct bert_ctx * ctx
]
def bert_n_max_tokens(ctx):
    return _lib.bert_n_max_tokens(ctx)

# now we are free
_lib.bert_free.argtypes = [
    ctypes.c_void_p # struct bert_ctx * ctx
]
def bert_free(ctx):
    return _lib.bert_free(ctx)

# tokenize
_lib.bert_tokenize_c.restype = ctypes.c_uint64
_lib.bert_tokenize_c.argtypes = [
    ctypes.c_void_p,                # struct bert_ctx * ctx
    ctypes.c_char_p,                # const char * text
    ctypes.POINTER(ctypes.c_int32), # int32_t * output
    ctypes.c_uint64,                # uint64_t n_max_tokens
]
def bert_tokenize(ctx, text, n_max_tokens):
    tokens = np.zeros(n_max_tokens, dtype=np.int32)
    tokens_p = tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    n_tokens = _lib.bert_tokenize_c(ctx, text.encode('utf-8'), tokens_p, n_max_tokens)
    return tokens[:n_tokens]

# detokenize
_lib.bert_detokenize_c.restype = ctypes.c_uint64
_lib.bert_detokenize_c.argtypes = [
    ctypes.c_void_p,                # struct bert_ctx * ctx
    ctypes.POINTER(ctypes.c_int32), # int32_t * tokens
    ctypes.c_char_p,                # char * text
    ctypes.c_uint64,                # uint64_t n_input
    ctypes.c_uint64,                # uint64_t n_output
    ctypes.c_bool,                  # bool debug
]
def bert_detokenize(ctx, tokens, max_len, debug):
    n_input = len(tokens)
    tokens = np.asarray(tokens, dtype=np.int32)
    output = ctypes.create_string_buffer(max_len)
    tokens_p = tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    _lib.bert_detokenize_c(ctx, tokens_p, output, n_input, max_len, debug)
    return output.value.decode('utf-8')

# encode batch
_lib.bert_encode_batch_c.argtypes = [
    ctypes.c_void_p,                 # struct bert_ctx * ctx
    ctypes.POINTER(ctypes.c_char_p), # const char ** texts
    ctypes.POINTER(ctypes.c_float),  # float * embeddings
    ctypes.c_int32,                  # int32_t n_inputs
    ctypes.c_bool,                   # bool normalize
    ctypes.c_int32,                  # int32_t n_threads
]
def bert_encode_batch_c(ctx, texts, embed_p, normalize, n_threads):
    n_inputs = len(texts)
    strings = (ctypes.c_char_p * n_inputs)()
    for j, s in enumerate(texts):
        strings[j] = s.encode('utf-8')
    return _lib.bert_encode_batch_c(ctx, strings, embed_p, n_inputs, normalize, n_threads)
def bert_encode_batch(ctx, texts, normalize, n_threads):
    n_inputs = len(texts)
    n_embd = bert_n_embd(ctx)
    embed = np.zeros((n_inputs, n_embd), dtype=np.float32)
    embed_p = embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    bert_encode_batch_c(ctx, texts, embed_p, normalize, n_threads)
    return embed
