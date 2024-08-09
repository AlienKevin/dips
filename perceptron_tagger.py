import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
import random
from tqdm import tqdm


class TaggerDataset(IterableDataset):
    def __init__(self, data, window_size=5, vocab_threshold=0.999, vocab=None, tagset=None):
        self.data = data
        self.window_size = window_size
        self.vocab_threshold = vocab_threshold
        if vocab is None:
            self.vocab = self.build_vocab()
        else:
            self.vocab = vocab
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
                unk_count += count
                continue
            vocab[char] = len(vocab)
        if '�' not in vocab:
            vocab['�'] = len(vocab)
        print(f"Number of characters classified as �: {unk_count}")
        vocab_size = len(vocab)
        one_hot_vocab = {char: torch.nn.functional.one_hot(torch.tensor(idx), num_classes=vocab_size).to(torch.float) for char, idx in vocab.items()}
        return one_hot_vocab

    def build_tagset(self):
        tagset = set()
        for sentence, tags in self.data:
            for tag in tags:
                tagset.add(tag)
        tagset = sorted(list(tagset))
        tagset = {tag: idx for idx, tag in enumerate(tagset)}
        tagset_size = len(tagset)
        print(tagset)
        one_hot_tagset = {tag: torch.nn.functional.one_hot(torch.tensor(idx), num_classes=tagset_size).to(torch.float) for tag, idx in tagset.items()}
        return one_hot_tagset

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
                window = [self.vocab.get(char, self.vocab['�']) for char in window]
                if len(window) < self.window_size:
                    pad_left = (self.window_size - len(window)) // 2
                    pad_right = self.window_size - len(window) - pad_left
                    pad_vector = torch.zeros(len(self.vocab))
                    window = [pad_vector] * pad_left + window + [pad_vector] * pad_right
                X = torch.cat(window)
                y = self.tagset[tags[i]]
                yield (X, y)

    def __len__(self):
        return self.num_windows

    def __iter__(self):
        return self.prepare_windows()


class Tagger(nn.Module):
    def __init__(self, vocab, tagset, window_size=5):
        super(Tagger, self).__init__()
        self.vocab = vocab
        self.tagset = tagset
        self.window_size = window_size
        self.linear = nn.Linear(len(vocab) * window_size, len(tagset))

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.linear(x)
    
    def tag(self, text, device):
        tags = []
        for i in range(len(text)):
            start = max(0, i - self.window_size // 2)
            end = i + self.window_size // 2 + 1
            window = text[start:end]
            window = [self.vocab.get(char, self.vocab['�']) for char in window]
            if len(window) < self.window_size:
                pad_left = (self.window_size - len(window)) // 2
                pad_right = self.window_size - len(window) - pad_left
                pad_vector = torch.zeros(len(self.vocab))
                window = [pad_vector] * pad_left + window + [pad_vector] * pad_right
            X = torch.cat(window)
            # Add a batch dimension
            outputs = self(X.to(device).unsqueeze(0))
            _, predicted = torch.max(outputs, 1)
            tag = list(self.tagset.keys())[predicted.item()]
            tags.append(((text[i] if text[i] in self.vocab else '�'), tag))
        return tags


def train_model(model, model_name, train_loader, validation_loader, criterion, optimizer, num_epochs, device):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X, y in tqdm(train_loader, total=len(train_loader)):
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), y.view(-1, y.shape[-1]))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_loss}")

        # Calculate validation loss
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for X_val, y_val in validation_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                outputs_val = model(X_val)
                loss_val = criterion(outputs_val.view(-1, outputs_val.shape[-1]), y_val.view(-1, y_val.shape[-1]))
                validation_loss += loss_val.item()
        avg_validation_loss = validation_loss / len(validation_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_validation_loss}")

        # Save the best model based on validation loss
        if avg_validation_loss < best_loss:
            best_loss = avg_validation_loss
            torch.save(model, f"{model_name}.pth")


def load_helper(dataset):
    label_names = dataset.features["pos_tags_ud"].feature.names
    id2label = { i:k for i, k in enumerate(label_names) }

    tagged_corpus = []
    for item in dataset:
        tokens = item['tokens']
        tags = [id2label[tag] for tag in item['pos_tags_ud']]
        char_tokens = []
        char_tags = []
        for token, tag in zip(tokens, tags):
            for i, char in enumerate(token):
                char_tokens.append(char)
                if i == 0:
                    char_tags.append(f"B-{tag}")
                else:
                    char_tags.append(f"I-{tag}")
        tagged_corpus.append((char_tokens, char_tags))
    return tagged_corpus


def load_hkcancor():
    dataset = load_dataset('nanyang-technological-university-singapore/hkcancor', split='train')
    return load_helper(dataset)


def load_cc100():
    dataset = load_dataset('AlienKevin/cc100-yue-tagged', split='train')
    return load_helper(dataset)


def train(model_name, training_dataset, batch_size, device):
    if training_dataset == 'hkcancor':
        tagged_corpus = load_hkcancor()
    elif training_dataset == 'cc100':
        tagged_corpus = load_cc100()

    print(tagged_corpus[0])

    random.seed(42)
    random.shuffle(tagged_corpus)

    validation_dataset = tagged_corpus[:100]
    train_dataset = tagged_corpus[100:]

    window_size = 5

    train_dataset = TaggerDataset(train_dataset, window_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    print('Training dataset vocab size:', len(train_dataset.vocab))
    print('Training dataset tagset size:', len(train_dataset.tagset))

    validation_dataset = TaggerDataset(validation_dataset, window_size, vocab=train_dataset.vocab, tagset=train_dataset.tagset)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    
    model = Tagger(train_dataset.vocab, train_dataset.tagset, window_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, model_name, train_loader, validation_loader, criterion, optimizer, num_epochs=5, device=device)


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


def load_ud_yue():
    from datasets import load_dataset

    # Load the universal_dependencies dataset from Hugging Face
    dataset = load_dataset('universal-dependencies/universal_dependencies', 'yue_hk', trust_remote_code=True)

    # Gather all word segmented utterances
    utterances = [[(token, upos_id_to_str(pos)) for token, pos in zip(sentence['tokens'], sentence['upos'])] for sentence in dataset['test']]

    return utterances


def merge_tokens(tagged_characters):
    merged_tokens = []
    current_token = []
    current_tag = None

    for char, tag in tagged_characters:
        if tag.startswith('B-'):
            if current_token:
                merged_tokens.append((''.join(current_token), current_tag))
            current_token = [char]
            current_tag = tag[2:]
        elif tag.startswith('I-'):
            if current_tag is None:
                print(f"Error: I-tag '{tag}' without preceding B-tag. Treating as B-tag.")
                current_token = [char]
                current_tag = tag[2:]
            elif tag[2:] != current_tag:
                print(f"Error: I-tag '{tag}' does not match current B-tag '{current_tag}'. Overwriting with B-tag.")
                current_token.append(char)
            else:
                current_token.append(char)
        else:
            if current_token:
                merged_tokens.append((''.join(current_token), current_tag))
                current_token = []
                current_tag = None
            merged_tokens.append((char, tag))

    if current_token:
        merged_tokens.append((''.join(current_token), current_tag))

    return merged_tokens

def test(model_name, device):
    from spacy.training import Example
    from spacy.scorer import Scorer
    from spacy.tokens import Doc
    from spacy.vocab import Vocab
    import json

    test_dataset = load_ud_yue()

    random.seed(42)
    random.shuffle(test_dataset)

    test_dataset = test_dataset[:100]

    model = torch.load(f"{model_name}.pth", weights_only=False)
    model.eval()

    V = Vocab()
    examples = []
    errors = []
    for reference in tqdm(test_dataset):
        hypothesis = merge_tokens(model.tag(''.join(token for token, _ in reference), device))
        reference_tokens = [''.join(char if char in model.vocab.keys() else '�' for char in token) for token, _ in reference]
        target = Doc(V, words=reference_tokens, spaces=[False for _ in reference], pos=[tag for _, tag in reference])
        predicted_doc = Doc(V, words=[token for token, _ in hypothesis], spaces=[False for _ in hypothesis], pos=[tag for _, tag in hypothesis])
        example = Example(predicted_doc, target)
        examples.append(example)

        if reference != hypothesis:
            errors.append({'reference': reference, 'hypothesis': hypothesis})

    scorer = Scorer()
    results = scorer.score(examples)

    with open('errors.jsonl', 'w') as f:
        for error in errors:
            f.write(json.dumps(error, ensure_ascii=False) + '\n')

    print(f"POS Tagging Accuracy: {results['pos_acc']}")
    print(f"Token F1 Score: {results['token_f']}")
    print(f"Token Precision: {results['token_p']}")
    print(f"Token Recall: {results['token_r']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train or test the POS tagger model.')
    parser.add_argument('--mode', choices=['train', 'test'], help='Mode to run the script in: train or test')
    parser.add_argument('--training_dataset', choices=['hkcancor', 'cc100'], help='Training dataset to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model_name = f"{args.training_dataset}_pos_perceptron_window_5_one_hot"

    if args.mode == 'train':
        train(model_name, args.training_dataset, args.batch_size, device=device)
    elif args.mode == 'test':
        test(model_name, device)
