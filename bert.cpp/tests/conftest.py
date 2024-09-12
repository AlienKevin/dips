##
## testing fixtures
##

import pytest

from transformers import AutoTokenizer
from bert_cpp import BertModel

TESTS_DIR = 'tests'
MODEL_DIR = 'models'

MODEL_LIST = [
    ('BAAI/bge-base-en-v1.5', 16),
    ('BAAI/bge-base-zh-v1.5', 16),
    ('BAAI/bge-m3', 16)
]

SIMPLE_STRINGS = [
    'hello world',
    'this is a test',
    'how are you ?',
]

TEST_STRINGS = [
    line.rstrip('\n') for line in open(f'{TESTS_DIR}/test_strings.txt')
]

@pytest.fixture(params=MODEL_LIST)
def tokenizers(request):
    repo_id, prec = request.param
    _, model_id = repo_id.split('/')
    model = BertModel(f'{MODEL_DIR}/{model_id}-f{prec}.gguf', allocate=False)
    tokhf = AutoTokenizer.from_pretrained(repo_id)
    return model, tokhf

@pytest.fixture(params=SIMPLE_STRINGS)
def simple_text(request):
    return request.param

@pytest.fixture(params=TEST_STRINGS)
def text(request):
    return request.param
