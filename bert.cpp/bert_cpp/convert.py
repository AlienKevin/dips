import sys
import torch

from gguf import GGUFWriter, GGMLQuantizationType
from transformers import AutoModelForTokenClassification, AutoTokenizer

KEY_PAD_ID = 'tokenizer.ggml.padding_token_id'
KEY_UNK_ID = 'tokenizer.ggml.unknown_token_id'
KEY_BOS_ID = 'tokenizer.ggml.bos_token_id'
KEY_EOS_ID = 'tokenizer.ggml.eos_token_id'
KEY_WORD_PREFIX = 'tokenizer.ggml.word_prefix'
KEY_SUBWORD_PREFIX = 'tokenizer.ggml.subword_prefix'

def convert_hf(repo_id, output_path, float_type='f16'):
    # convert to ggml quantization type
    if float_type not in ['f16', 'f32']:
        print(f'Float type must be f16 or f32, got: {float_type}')
        sys.exit(1)
    else:
        qtype = GGMLQuantizationType[float_type.upper()]
        dtype0 = {'f16': torch.float16, 'f32': torch.float32}[float_type]

    # load tokenizer and model
    vocab = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForTokenClassification.from_pretrained(repo_id)
    config = model.config

    # get token list
    token_list = vocab.convert_ids_to_tokens(range(vocab.vocab_size))

    # detect subword scheme
    if getattr(vocab.backend_tokenizer.pre_tokenizer, 'replacement', None) == '▁':
        word_prefix = '▁'
        subword_prefix = ''
    else:
        word_prefix = ''
        subword_prefix = '##'

    # print model
    param_keys = [
        'vocab_size', 'max_position_embeddings', 'hidden_size', 'embedding_size', 'intermediate_size',
        'num_attention_heads', 'num_hidden_layers', 'layer_norm_eps'
    ]
    print('PARAMS')
    for k in param_keys:
        v = getattr(config, k)
        print(f'{k:<24s} = {v}')
    print()

    # print vocab
    vocab_keys = [
        'vocab_size', 'pad_token_id', 'unk_token_id', 'cls_token_id', 'sep_token_id'
    ]
    print('VOCAB')
    for k in vocab_keys:
        v = getattr(vocab, k)
        print(f'{k:24s} = {v}')
    print(f'{"word_prefix":24s} = {word_prefix}')
    print(f'{"subword_prefix":24s} = {subword_prefix}')
    print()

    # start to write GGUF file
    gguf_writer = GGUFWriter(output_path, 'bert')

    # write metadata
    gguf_writer.add_name('BERT')
    gguf_writer.add_description('GGML BERT model')
    gguf_writer.add_file_type(qtype)

    # write model params
    gguf_writer.add_uint32('vocab_size', config.vocab_size)
    gguf_writer.add_uint32('max_position_embedding', config.max_position_embeddings)
    gguf_writer.add_uint32('hidden_size', config.hidden_size)
    gguf_writer.add_uint32('embedding_size', config.embedding_size)
    gguf_writer.add_uint32('intermediate_size', config.intermediate_size)
    gguf_writer.add_uint32('num_attention_heads', config.num_attention_heads)
    gguf_writer.add_uint32('num_hidden_layers', config.num_hidden_layers)
    gguf_writer.add_float32('layer_norm_eps', config.layer_norm_eps)

    # write vocab params
    gguf_writer.add_int32(KEY_PAD_ID, vocab.pad_token_id)
    gguf_writer.add_int32(KEY_UNK_ID, vocab.unk_token_id)
    gguf_writer.add_int32(KEY_BOS_ID, vocab.cls_token_id)
    gguf_writer.add_int32(KEY_EOS_ID, vocab.sep_token_id)
    gguf_writer.add_string(KEY_WORD_PREFIX, word_prefix)
    gguf_writer.add_string(KEY_SUBWORD_PREFIX, subword_prefix)
    gguf_writer.add_token_list(token_list)

    # write tensors
    print('TENSORS')
    for name, data in model.state_dict().items():
        name = name.removeprefix('electra.')
        # get correct dtype
        if 'LayerNorm' in name or 'bias' in name:
            dtype = torch.float32
        else:
            dtype = dtype0

        # print info
        shape_str = str(list(data.shape))
        print(f'{name:64s} = {shape_str:16s} {data.dtype} → {dtype}')

        # do conversion
        data = data.to(dtype)

        # add to gguf output
        gguf_writer.add_tensor(name, data.numpy())

    # execute and close writer
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    # print success
    print()
    print(f'GGML model written to {output_path}')

# script usage
if __name__ == '__main__':
    # primay usage
    if len(sys.argv) < 3:
        print('Usage: convert-to-ggml.py repo_id output_path [float-type=f16,f32]\n')
        sys.exit(1)

    # output in the same directory as the model
    repo_id = sys.argv[1]
    output_path = sys.argv[2]

    # get float type
    if len(sys.argv) > 3:
        kwargs = {'float_type': sys.argv[3].lower()}
    else:
        kwargs = {}

    # convert to ggml
    convert_hf(repo_id, output_path, **kwargs)
