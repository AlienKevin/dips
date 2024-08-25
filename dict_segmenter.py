import argparse
from datasets import load_dataset
from collections import Counter
from utils import normalize, score_tags
from tagger_dataset import load_tagged_dataset

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def longest_prefix(self, string):
        node = self.root
        longest = ''
        for i, char in enumerate(string):
            if char not in node.children:
                break
            node = node.children[char]
            if node.is_end:
                longest = string[:i+1]
        return longest

def gather_tokens_from_dataset(dataset_name):
    dataset = load_dataset(f'AlienKevin/{dataset_name}', split="train")
    token_counter = Counter()
    for example in dataset:
        tokens = [normalize(token) for token in example['tokens']]
        token_counter.update(tokens)
    unique_tokens = sorted(token_counter.keys())
    return unique_tokens

def train(args):
    all_tokens = set()
    for dataset_name in args.train_datasets:
        tokens = gather_tokens_from_dataset(dataset_name)
        all_tokens.update(tokens)
        print(f"Total unique tokens in {dataset_name}: {len(tokens)}")
        print(f"First 10 tokens: {tokens[:10]}")
        print(f"Last 10 tokens: {tokens[-10:]}")
    
    all_tokens = sorted(list(all_tokens))
    output_file = f"data/{'_'.join(dataset.removesuffix('-seg').removesuffix('-tagged') for dataset in args.train_datasets)}_vocab.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for token in all_tokens:
            f.write(f"{token}\n")
    print(f"Combined vocabulary has been written to {output_file}")
    print(f"Total unique tokens across all datasets: {len(all_tokens)}")

def segment(text, trie):
    segmented = []
    i = 0
    while i < len(text):
        longest_match = trie.longest_prefix(text[i:])
        if not longest_match:
            longest_match = text[i]
        segmented.append(longest_match)
        i += len(longest_match)
    return segmented

def load_vocab(vocab_file):
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f]
    trie = Trie()
    for word in vocab:
        trie.insert(word)
    return trie

def test(args):
    test_dataset = load_tagged_dataset(args.test_dataset, 'test')
    trie = load_vocab(f"data/{'_'.join(dataset.removesuffix('-seg').removesuffix('-tagged') for dataset in args.train_datasets)}_vocab.txt")
    results, errors = score_tags(test_dataset, lambda text: [(token, 'X') for token in segment(text, trie)])
    import json
    with open('errors.jsonl', 'w', encoding='utf-8') as f:
        for error in errors:
            json.dump({"REF": ' '.join(token for token, _ in error['reference']), "HYP": ' '.join(token for token, _ in error['hypothesis'])}, f, ensure_ascii=False)
            f.write('\n')
    print(f"Errors have been written to errors.jsonl")
    print(f"Token F1 Score: {results['token_f']}")
    print(f"Token Precision: {results['token_p']}")
    print(f"Token Recall: {results['token_r']}")


def main():
    parser = argparse.ArgumentParser(description="Extract vocabulary from a dataset")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help="Mode of operation")
    parser.add_argument("--train_datasets", type=str, nargs='+', required=True, help="Name(s) of the dataset(s) to use for training")
    parser.add_argument("--test_dataset", type=str, required=True, help="Name of the dataset to use for testing")
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)


if __name__ == "__main__":
    main()
