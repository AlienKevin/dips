from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import os
import datasets
from collections import Counter
import opencc

# load model and tokenizer
model_id = "electra_small_layers_6_multi"

output_dir = model_id + "_compressed"
os.makedirs(output_dir, exist_ok=True)

feature = "token-classification"
model = AutoModelForTokenClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)


def get_chars_from_dataset(dataset_name, field_name='chars'):
    dataset = datasets.load_dataset(dataset_name, split='train')
    all_chars = set()
    for chars in dataset[field_name]:
        all_chars.update(chars)
    return all_chars

def get_top_chars(dataset_name, field_name='chars', coverage=0.9999):
    dataset = datasets.load_dataset(dataset_name, split='train')
    all_chars = []
    for chars in dataset[field_name]:
        all_chars.extend(chars)
    char_counts = Counter(all_chars)
    total_chars = sum(char_counts.values())
    
    sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
    
    top_chars = set()
    cumulative_count = 0
    for char, count in sorted_chars:
        top_chars.add(char)
        cumulative_count += count
        if cumulative_count / total_chars >= coverage:
            break
    
    return top_chars

# Collect characters from HKCanCor
hkcancor_chars = get_chars_from_dataset('AlienKevin/hkcancor-multi')

# Collect top characters from Wiki Yue
wiki_yue_chars = get_top_chars('AlienKevin/wiki-yue-long-multi')

# Combine the character sets
common_chars = hkcancor_chars.union(wiki_yue_chars)

# Convert to simplified Chinese
converter = opencc.OpenCC('t2s.json')
simplified_chars = set(converter.convert(''.join(common_chars)))

# Add simplified characters to the collection
common_chars = common_chars.union(simplified_chars)

print(f"Total number of common Chinese characters: {len(common_chars)}")


def is_chinese_char(char):
    """Check if a character is a Chinese character."""
    import re
    pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002ebef\U00030000-\U000323af\ufa0e\ufa0f\ufa11\ufa13\ufa14\ufa1f\ufa21\ufa23\ufa24\ufa27\ufa28\ufa29\u3006\u3007]')
    return pattern.match(char) is not None and char in common_chars

def is_punctuation(char):
    """Check if a character is a Chinese or English punctuation."""
    chinese_punctuation = set(['！', '：', '；', '“', '”', '‘', '’', '【', '】', '（', '）',
            '「', '」', '﹁', '﹂', '『','』', '《', '》', '？', '，', '。', '、', '／', '＋',
            '〈','〉', '︿', '﹀', '［', '］', '‧',
            # Small Form Variants for Chinese National Standard CNS 11643
            '﹐', '﹑','﹒', '﹔', '﹕', '﹖', '﹗', '﹘', '﹙', '﹚', '﹛', '﹜', '﹝', '﹞', '﹟'])
    english_punctuation = set(['~', '`', '!',  '(', ')', '-', '_', '{', '}', '[', ']', '|', '\\', ':', ';', '"', '\'', '<', '>', ',', '.', '?', '/'])
    return char in chinese_punctuation or char in english_punctuation

def is_ascii(char):
    """Check if a character is an ASCII character."""
    return ord(char) < 128

def should_keep_token(token):
    """Determine if a token should be kept based on the criteria."""
    if token in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']:
        return True
    if len(token) == 1:
        char = token[0]
        return is_chinese_char(char) or is_punctuation(char) or is_ascii(char)
    return False

# Get the original vocabulary
original_vocab = tokenizer.get_vocab()

# Filter the vocabulary
filtered_vocab = {token for token in original_vocab.keys() if should_keep_token(token)}

# Create a new vocabulary with reassigned IDs
new_vocab = {token: idx for idx, token in enumerate(sorted(list(filtered_vocab)))}

# Create a new ElectraTokenizer with the filtered vocabulary
from transformers import ElectraTokenizerFast
import tempfile

# Create a temporary file to store the new vocabulary
with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as temp_vocab_file:
    for token in new_vocab.keys():
        temp_vocab_file.write(f"{token}\n")
    temp_vocab_path = temp_vocab_file.name

# Create a new tokenizer with the filtered vocabulary
tokenizer = ElectraTokenizerFast(vocab_file=temp_vocab_path)

# Clean up the temporary file
import os
os.unlink(temp_vocab_path)

# Update the model's embedding layer
old_embeddings = model.electra.embeddings.word_embeddings.weight.data
new_embeddings = torch.nn.Embedding(len(new_vocab), old_embeddings.size(1))

for token, new_idx in new_vocab.items():
    old_idx = original_vocab[token]
    new_embeddings.weight.data[new_idx] = old_embeddings[old_idx]

model.electra.embeddings.word_embeddings = new_embeddings
model.config.vocab_size = len(new_vocab)

print(f"Vocabulary size reduced from {len(original_vocab)} to {len(new_vocab)}")

# Save the model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")

# Convert model to fp16
import torch

# Convert model to fp16
model_fp16 = model.half()

# Save the fp16 model
model_fp16.save_pretrained(output_dir)

print(f"Model converted to fp16 and saved to {output_dir}")
