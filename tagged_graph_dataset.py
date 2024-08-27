import torch
from torch_geometric.data import Data, Dataset
from typing import List
import pygtrie
from vocab import Vocab
from utils import read_pretrained_embeddings
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt

class TaggedGraphDataset(Dataset):
    def __init__(self, tagged_dataset, lexicon: List[str], vocab_threshold: float = 0.9999, vocab=None):
        super().__init__()
        self.tagged_dataset = tagged_dataset
        self.trie = self._build_trie(lexicon)
        self.vocab_threshold = vocab_threshold
        if vocab is None:
            self.vocab = self.build_vocab()
        else:
            self.vocab = vocab

    def _build_trie(self, lexicon: List[str]) -> pygtrie.CharTrie:
        trie = pygtrie.CharTrie()
        for word in lexicon:
            trie[word] = True
        return trie

    def build_vocab(self):
        from collections import Counter
        vocab_counter = Counter()
        def count_chars(example):
            for char in example['tokens']:
                vocab_counter[char] += 1
            return None
        self.tagged_dataset.map(count_chars)
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
        print(f"Number of characters classified as [UNK]: {unk_count}")
        return Vocab(vocab)

    def _process_item(self, item) -> Data:
        chars = item['tokens']
        labels = item['tags']
        
        # Create node features (learnable dense vectors)
        x = torch.tensor([self.vocab[char] for char in chars], dtype=torch.long)
        
        # Create edges
        edge_index = set()
        for i in range(len(chars)):
            for j in range(i + 1, len(chars) + 1):
                substring = ''.join(chars[i:j])
                if self.trie.has_key(substring):
                    edge_index.add((i, j - 1))
                    edge_index.add((j - 1, i))
        
        # Add edges between all neighboring characters if not already added
        for i in range(len(chars) - 1):
            edge_index.add((i, i + 1))
            edge_index.add((i + 1, i))
        
        edge_index = torch.tensor(sorted(list(edge_index)), dtype=torch.long).t().contiguous()
        
        y = torch.tensor(labels, dtype=torch.long)
        
        # nx.draw(torch_geometric.utils.to_networkx(Data(x=x, edge_index=edge_index, y=y), to_undirected=True))
        # plt.show()

        return Data(x=x, edge_index=edge_index, y=y)

    def len(self) -> int:
        return len(self.tagged_dataset)

    def get(self, idx: int) -> Data:
        item = self.tagged_dataset[idx]
        return self._process_item(item)
