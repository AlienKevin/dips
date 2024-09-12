# benchmarking BERT

import re
import json
from .model import BertModel

# get batch indices
def batch_indices(length, batch_size):
    return [(i, min(i+batch_size, length)) for i in range(0, length, batch_size)]

# split text into chunks
def list_splitter(text, maxlen):
    for i, j in batch_indices(len(text), maxlen):
        yield text[i:j]

# default paragraph splitter
def text_splitter(text, delim, min_len=1, max_len=None):
    if delim is not None:
        paras = [p.strip() for p in re.split(delim, text)]
    else:
        paras = [text]
    paras = [p for p in paras if len(p) >= min_len]
    if max_len is not None:
        paras = list(chain.from_iterable(
            list_splitter(p, max_len) for p in paras
        ))
    return paras

# generate loader for jsonl file
def stream_jsonl(path, max_rows=None):
    with open(path) as fid:
        for i, line in enumerate(fid):
            if max_rows is not None and i >= max_rows:
                break
            yield json.loads(line)

# load column of jsonl file and chunkify
def load_jsonl(wiki_path, max_rows=1024, column='text', min_len=32, max_len=None):
    splitter = lambda s: text_splitter(s, '\n', min_len=min_len, max_len=max_len)
    stream = stream_jsonl(wiki_path, max_rows=max_rows)
    chunks = sum([splitter(d[column]) for d in stream], [])
    return chunks

# run benchmark for one model/data pair
def benchmark(model, data, min_len=32, max_len=None, batch_size=32, max_rows=None, columns='text', use_cpu=False):
    if type(model) is str:
        model = BertModel(model, batch_size=batch_size, use_cpu=use_cpu)
    if type(data) is str:
        data = load_jsonl(data, max_rows=max_rows, column=column, min_len=min_len, max_len=max_len)
    return model.embed(data)
