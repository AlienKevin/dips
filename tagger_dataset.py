import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
import datasets
from utils import normalize
from vocab import Vocab

class TaggerDataset(IterableDataset):
    def __init__(self, data, window_size, tag_context_size, vocab_threshold, sliding=True, vocab=None, tagset=None):
        self.sliding = sliding
        self.data = data
        self.window_size = window_size
        self.vocab_threshold = vocab_threshold
        if vocab is None:
            self.vocab = self.build_vocab()
        else:
            self.vocab = vocab
        self.tag_context_size = tag_context_size
        if tagset is None:
            self.tagset = self.build_tagset()
        else:
            self.tagset = tagset
        self.num_windows = self.calculate_num_windows()

    def build_vocab(self):
        from collections import Counter
        vocab_counter = Counter()
        def count_chars(example):
            for char in example['tokens']:
                vocab_counter[char] += 1
            return None
        self.data.map(count_chars)
        total_chars = sum(vocab_counter.values())
        sorted_vocab = sorted(vocab_counter.items(), key=lambda x: x[1], reverse=True)
        cumulative_count = 0
        vocab = {}
        unk_count = 0
        for char, count in sorted_vocab:
            cumulative_count += count
            if cumulative_count / total_chars > self.vocab_threshold:
                unk_count += 1
                continue
            vocab[char] = len(vocab)
        if '[UNK]' not in vocab:
            vocab['[UNK]'] = len(vocab)
        vocab['[PAD]'] = -100
        print(f"Number of characters classified as [UNK]: {unk_count}")
        return Vocab(vocab)

    def build_tagset(self):
        tagset = self.data.features['tags'].feature.names
        tagset = {tag: idx for idx, tag in enumerate(tagset)}
        return Vocab(tagset)

    def calculate_num_windows(self):
        num_windows = 0
        for example in self.data:
            tokens = example['tokens']
            tags = example['tags']
            num_windows += len(tokens)
        return num_windows

    def prepare_windows(self):
        for example in self.data:
            tokens = example['tokens']
            tags = example['tags']
            for i in range(len(tokens)):
                start = max(0, i - self.window_size // 2)
                end = i + self.window_size // 2 + 1
                window = tokens[start:end]
                window = [self.vocab[char] for char in window]
                if len(window) < self.window_size:
                    pad_left = (self.window_size - len(window)) // 2
                    pad_right = self.window_size - len(window) - pad_left
                    window = [self.vocab['[PAD]']] * pad_left + window + [self.vocab['[PAD]']] * pad_right
                X = torch.tensor(window)
                y = torch.tensor(tags[i])
                if self.tag_context_size > 0:
                    context_tags = tags[max(0, i - self.tag_context_size):i]
                    if len(context_tags) < self.tag_context_size:
                        pad_size = self.tag_context_size - len(context_tags)
                        context_tags = ['[PAD]'] * pad_size + context_tags
                    tag_context = torch.nn.functional.one_hot(torch.tensor(context_tags), num_classes=len(self.tagset)).to(torch.float)
                    yield (X, tag_context, y)
                else:
                    yield (X, None, y)

    def __len__(self):
        if self.sliding:
            return self.num_windows
        else:
            return len(self.data)

    def __iter__(self):
        if self.sliding:
            for window in self.prepare_windows():
                yield window
        else:
            for example in self.data:
                tokens = example['tokens']
                tags = example['tags']
                X = torch.tensor([self.vocab[char] for char in tokens])
                y = torch.tensor(tags)
                if 'logits' in example:
                    yield (X, y, torch.tensor(example['logits']))
                else:
                    yield (X, y)



def load_helper(dataset, tagging_scheme):
    def process_item(item, tagging_scheme):
        tokens = item['tokens']
        tags = item['tags']
        char_tokens = []
        char_tags = []
        for token, tag in zip(tokens, tags):
            token = normalize(token)
            if tag == '[PAD]':
                char_tokens.append(token)
                char_tags.append(tag)
                continue

            if tagging_scheme == 'BIES' and len(token) == 1:
                char_tokens.append(token)
                char_tags.append(f"S-{tag}")
                continue
            
            for i, char in enumerate(token):
                char_tokens.append(char)
                if i == 0:
                    char_tags.append(f"B-{tag}")
                elif tagging_scheme == 'BIES' and i == len(token) - 1:
                    char_tags.append(f"E-{tag}")
                else:
                    char_tags.append(f"I-{tag}")
        return {'tokens': char_tokens, 'tags': char_tags}

    tagged_dataset = dataset.map(
        lambda item: process_item(item, tagging_scheme),
        remove_columns=dataset.column_names,
        num_proc=20
    )
    return tagged_dataset


def load_ud(lang='yue',split='test'):
    # Load the universal_dependencies dataset from Hugging Face
    dataset = load_dataset('universal-dependencies/universal_dependencies', lang, trust_remote_code=True)

    # Gather all word segmented utterances
    utterances = [[(token, upos_id_to_str(pos)) for token, pos in zip(sentence['tokens'], sentence['upos'])] for sentence in dataset[split]]

    return utterances


def load_tagged_dataset(dataset_name, split, tagging_scheme=None, transform=None, output_format='follow_split', segmentation_only=False):
    dataset = load_dataset(f'AlienKevin/{dataset_name}' , split=split)

    if dataset_name.endswith('-multi'):
        dataset = dataset.rename_column('chars', 'tokens').rename_column('labels', 'tags')
        if split == 'test' or output_format == 'test':
            tag_label_names = dataset.features["tags"].feature.names
            tag_id2label = { i:k for i, k in enumerate(tag_label_names) }
            utterances = []
            for item in dataset:
                utterances.append([(token, tag_id2label[tag]) for token, tag in zip(item['tokens'], item['tags'])])
            return utterances
        else:
            return dataset

    dataset = dataset.select(range(min(len(dataset), 2000000)))

    segmentation_only = dataset_name.endswith('-seg') or segmentation_only

    if segmentation_only:
        # Remove all fields except 'tokens'
        columns_to_remove = [col for col in dataset.features if col != 'tokens']
        
        dataset = dataset.map(lambda example: {
            'tokens': example['tokens'],
            'pos_tags_ud': ['X' for _ in example['tokens']]
        }, features=datasets.Features({
            'tokens': datasets.Sequence(datasets.features.Value('string')),
            'pos_tags_ud': datasets.Sequence(datasets.features.ClassLabel(names=['X']))
        }), remove_columns=columns_to_remove, num_proc=20)

    tag_label_names = dataset.features["pos_tags_ud"].feature.names
    tag_id2label = { i:k for i, k in enumerate(tag_label_names) }

    dataset = dataset.map(lambda example: {
        'tokens': example['tokens'],
        'tags': [tag_id2label[tag] for tag in example['pos_tags_ud']]
    }, num_proc=20)
    
    if split == 'test' or output_format == 'test':
        utterances = []
        for item in dataset:
            utterances.append([(token, tag) for token, tag in zip(item['tokens'], item['tags'])])

        return utterances
    else:
        dataset = load_helper(dataset, tagging_scheme)

        # Define the tag set based on segmentation_only and tagging_scheme
        if segmentation_only:
            tag_names = [f"{letter}-X" for letter in tagging_scheme]
        else:
            tag_names = [f"{letter}-{tag}" for letter in tagging_scheme for tag in tag_label_names]

        # Create a new feature for tags using ClassLabel
        tag_feature = datasets.Sequence(datasets.features.ClassLabel(names=tag_names))

        if transform:
            def apply_transform(example):
                chars, tags = example['tokens'], example['tags']
                transformed_chars, transformed_tags = zip(*((transformed_token, transformed_tag) 
                                    for token, tag in zip(chars, tags)
                                    for transformed_token, transformed_tag in transform(token, tag)))
                return {'tokens': list(transformed_chars), 'tags': tag_feature.feature.str2int(list(transformed_tags))}

            dataset = dataset.map(apply_transform, num_proc=20, features=datasets.Features({
                'tokens': datasets.Sequence(datasets.Value('string')),
                'tags': tag_feature
            }))
        else:
            dataset = dataset.cast_column('tags', tag_feature)

        return dataset


def upos_id_to_str(upos_id):
    names=[
        "NOUN",
        "PUNCT",
        "ADP",
        "NUM",
        "SYM",
        "SCONJ",
        "ADJ",
        "PART",
        "DET",
        "CCONJ",
        "PROPN",
        "PRON",
        "X",
        "X",
        "ADV",
        "INTJ",
        "VERB",
        "AUX",
    ]
    return names[upos_id]
