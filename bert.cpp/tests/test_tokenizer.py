##
## tokenizer test suite
##

import pytest

def test_tokenizer_loop(tokenizers, simple_text):
    model, _ = tokenizers
    tokens = model.tokenize(simple_text)
    text = model.detokenize(tokens)
    assert text == simple_text

def test_tokenizer_hf(tokenizers, text):
    model, tokhf = tokenizers
    toks_bert = model.tokenize(text)
    toks_hf = tokhf.encode(text)
    assert toks_bert.tolist() == toks_hf
