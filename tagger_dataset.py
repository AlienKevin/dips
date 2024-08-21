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
        for sentence, tags in self.data:
            for char in sentence:
                vocab_counter[char] += 1
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
        vocab['[PAD]'] = len(vocab)
        print(f"Number of characters classified as [UNK]: {unk_count}")
        return Vocab(vocab)

    def build_tagset(self):
        tagset = set()
        for sentence, tags in self.data:
            for tag in tags:
                tagset.add(tag)
        if self.tag_context_size > 0 or not self.sliding:
            tagset.add('[PAD]')
        tagset = sorted(list(tagset))
        tagset = {tag: idx for idx, tag in enumerate(tagset)}
        return Vocab(tagset)

    def calculate_num_windows(self):
        num_windows = 0
        for sentence, tags in self.data:
            num_windows += len(sentence)
        return num_windows

    def prepare_windows(self):
        for sentence, tags in self.data:
            for i in range(len(sentence)):
                start = max(0, i - self.window_size // 2)
                end = i + self.window_size // 2 + 1
                window = sentence[start:end]
                window = [self.vocab[char] for char in window]
                if len(window) < self.window_size:
                    pad_left = (self.window_size - len(window)) // 2
                    pad_right = self.window_size - len(window) - pad_left
                    window = [self.vocab['[PAD]']] * pad_left + window + [self.vocab['[PAD]']] * pad_right
                X = torch.tensor(window)
                y = torch.tensor(self.tagset[tags[i]])
                if self.tag_context_size > 0:
                    context_tags = tags[max(0, i - self.tag_context_size):i]
                    if len(context_tags) < self.tag_context_size:
                        pad_size = self.tag_context_size - len(context_tags)
                        context_tags = ['[PAD]'] * pad_size + context_tags
                    tag_context = torch.nn.functional.one_hot(torch.tensor([self.tagset[tag] for tag in context_tags]), num_classes=len(self.tagset)).to(torch.float)
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
            for sentence, tags in self.data:
                X = torch.tensor([self.vocab[char] for char in sentence])
                y = torch.tensor([self.tagset[tag] for tag in tags])
                yield (X, y)



def load_helper(dataset, tagging_scheme):
    tagged_corpus = []
    for item in dataset:
        tokens = item['tokens']
        tags = item['tags']
        char_tokens = []
        char_tags = []
        for token, tag in zip(tokens, tags):
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
        tagged_corpus.append((char_tokens, char_tags))
    return tagged_corpus


def load_ud(lang='yue',split='test'):
    # Load the universal_dependencies dataset from Hugging Face
    dataset = load_dataset('universal-dependencies/universal_dependencies', lang, trust_remote_code=True)

    # Gather all word segmented utterances
    utterances = [[(token, upos_id_to_str(pos)) for token, pos in zip(sentence['tokens'], sentence['upos'])] for sentence in dataset[split]]

    return utterances


def load_tagged_dataset(dataset_name, split, tagging_scheme=None, transform=None, output_format='follow_split'):
    dataset = load_dataset(f'AlienKevin/{(f"{dataset_name}-tagged") if dataset_name not in ["ctb8", "msr-seg", "as-seg", "cityu-seg", "pku-seg"] else dataset_name}' , split=split)

    if dataset_name.endswith('-seg'):
        dataset = dataset.map(lambda example: {
            'tokens': example['tokens'],
            'pos_tags_ud': ['X' for _ in example['tokens']]
        }, features=datasets.Features({
            'tokens': datasets.Sequence(datasets.features.Value('string')),
            'pos_tags_ud': datasets.Sequence(datasets.features.ClassLabel(names=['X']))
        }))

    tab_label_names = dataset.features["pos_tags_ud"].feature.names
    tag_id2label = { i:k for i, k in enumerate(tab_label_names) }

    dataset = dataset.map(lambda example: {
        'tokens': example['tokens'],
        'tags': [tag_id2label[tag] for tag in example['pos_tags_ud']]
    })
    
    if split == 'test' or output_format == 'test':
        utterances = []
        for item in dataset:
            utterances.append([(token, tag) for token, tag in zip(item['tokens'], item['tags'])])

        return utterances
    else:
        dataset = load_helper(dataset, tagging_scheme)
        if transform:
            transformed_dataset = []
            for chars, tags in dataset:
                transformed_chars, transformed_tags = zip(*((transformed_token, transformed_tag) for token, tag in zip(chars, tags)
                                    for transformed_token, transformed_tag in transform(normalize(token), tag)))
                transformed_dataset.append((transformed_chars, transformed_tags))
            dataset = transformed_dataset
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
