import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
import random
from tqdm import tqdm
import math


class TaggerDataset(IterableDataset):
    def __init__(self, data, window_size, tag_context_size, vocab_threshold, vocab=None, tagset=None):
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
                unk_count += count
                continue
            vocab[char] = len(vocab)
        if '[UNK]' not in vocab:
            vocab['[UNK]'] = len(vocab)
        vocab['[PAD]'] = len(vocab)
        print(f"Number of characters classified as [UNK]: {unk_count}")
        return vocab

    def build_tagset(self):
        tagset = set()
        for sentence, tags in self.data:
            for tag in tags:
                tagset.add(tag)
        if self.tag_context_size > 0:
            tagset.add('[PAD]')
        tagset = sorted(list(tagset))
        tagset = {tag: idx for idx, tag in enumerate(tagset)}
        return tagset

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
                window = [self.vocab.get(char, self.vocab['[UNK]']) for char in window]
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
                    tag_context = tag_context.view(-1)  # Flatten the logits
                    yield (X, tag_context, y)
                else:
                    yield (X, None, y)

    def __len__(self):
        return self.num_windows

    def __iter__(self):
        return self.prepare_windows()


class LanguageModel:
    def __init__(self, texts, vocab, ngrams):
        print(texts[0])
        print(vocab)
        self.texts = texts
        self.vocab = vocab
        self.ngrams = ngrams
        self.model = self.build_ngram_lm()
    
    def get_pad_token(self):
        return self.vocab['[PAD]'] if '[PAD]' in self.vocab else len(self.vocab)

    def pad_and_lookup(self, text):
        pad_token = self.get_pad_token()
        fallback = self.vocab['[UNK]'] if '[UNK]' in self.vocab else pad_token
        return [pad_token] * self.ngrams + [self.vocab.get(char, fallback) for char in text] + [pad_token] * self.ngrams

    def build_ngram_lm(self):
        from collections import defaultdict
        ngram_counts = defaultdict(lambda: defaultdict(int))
        total_counts = defaultdict(int)

        for text in tqdm(self.texts, desc="Building n-gram LM"):
            padded_text = self.pad_and_lookup(text)
            for i in range(len(padded_text) - self.ngrams):
                ngram = tuple(padded_text[i:i + self.ngrams])
                next_char = padded_text[i + self.ngrams]
                ngram_counts[ngram][next_char] += 1
                total_counts[ngram] += 1

        vocab_size = len(self.vocab)
        ngram_probabilities = {ngram: {token: (count + 1) / (total_counts[ngram] + vocab_size) for token, count in next_tokens.items()} for ngram, next_tokens in ngram_counts.items()}
        return ngram_probabilities

    def score(self, text):
        padded_text = self.pad_and_lookup(text)
        score = 1.0
        for i in range(len(padded_text) - self.ngrams):
            ngram = tuple(padded_text[i:i + self.ngrams])
            next_char = padded_text[i + self.ngrams]
            score *= self.score_token(ngram, next_char)
        return score

    def score_token(self, prefix, token):
        if len(prefix) < self.ngrams:
            ngram = tuple([self.get_pad_token()] * (self.ngrams - len(prefix)) + prefix)
        else:
            ngram = tuple(prefix)
        if ngram in self.model and token in self.model[ngram]:
            return self.model[ngram][token]
        elif ngram in self.model:
            return 1 / (len(self.vocab) + sum(self.model[ngram].values()))  # Add-1 smoothing for unseen ngrams
        else:
            return 1 / len(self.vocab)  # Handle case where ngram is not in model

class Tagger(nn.Module):
    def __init__(self, vocab, tagset, window_size, tag_context_size, embedding_type, embedding_dim, autoregressive_scheme, network_depth):
        super(Tagger, self).__init__()
        self.vocab = vocab
        self.tagset = tagset
        self.window_size = window_size
        self.embedding_type = embedding_type
        self.autoregressive_scheme = autoregressive_scheme
        self.tag_context_size = tag_context_size
        if embedding_type == 'one_hot':
            embedding_dim = len(vocab)
            self.embedding = nn.Embedding.from_pretrained(torch.eye(embedding_dim).to(torch.float), freeze=True)
        else:
            self.embedding = nn.Embedding(len(vocab), embedding_dim)
        if network_depth == 2:
            self.linear = nn.Sequential(
                nn.Linear(embedding_dim * window_size + (tag_context_size * len(self.tagset) if self.autoregressive_scheme else 0), embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, len(tagset))
            )
        elif network_depth == 1:
            self.linear = nn.Linear(embedding_dim * window_size + (tag_context_size * len(self.tagset) if self.autoregressive_scheme else 0), len(tagset))
        else:
            raise ValueError(f"Invalid network depth: {network_depth}")

    def get_embedding(self, char):
        if char not in self.vocab:
            char = '[UNK]'
        return self.embedding(torch.tensor(self.vocab[char]).to(device))

    def forward(self, x, extra_logits=None):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)  # Flatten the input
        if extra_logits is not None:
            x = torch.cat([x, extra_logits], dim=1)
        return self.linear(x)

    def decode(self, text, pos_lm, beam_size, device):
        pad_tag_index = self.tagset['[PAD]']
        pad_one_hot = torch.nn.functional.one_hot(torch.tensor(pad_tag_index), num_classes=len(self.tagset)).float()
        previous_predictions = torch.stack([pad_one_hot] * self.tag_context_size)
        if pos_lm is None:
            # Greedy decoding
            tag_ids = []
            for i in range(len(text)):
                scores = self.get_scores(text, i, previous_predictions, device)
                _, predicted_tag = torch.max(scores, dim=1)
                tag_ids.append(predicted_tag.item())
                if self.autoregressive_scheme:
                    if self.autoregressive_scheme == 'teacher_forcing':
                        one_hot_prediction = torch.nn.functional.one_hot(predicted_tag, num_classes=len(self.tagset)).to(torch.float).to(device)
                        previous_predictions = self.update_previous_predictions(previous_predictions.to(device), one_hot_prediction)
                    else:
                        previous_predictions = self.update_previous_predictions(previous_predictions.to(device), scores)
        else:
            # Viterbi decoding with beam search incorporating pos_lm.score
            beam = [(0, [], previous_predictions)]  # (score, tag_sequence, previous_predictions)
            for i in range(len(text)):
                new_beam = []
                for score, tag_sequence, prev_preds in beam:
                    scores = self.get_scores(text, i, prev_preds, device)
                    scores = torch.log_softmax(scores, dim=1).view(-1)
                    for tag, tag_score in enumerate(scores):
                        new_score = score + tag_score.item()
                        lm_score = pos_lm.score_token(tag_sequence, tag)
                        combined_score = new_score + math.log(lm_score)
                        new_tag_sequence = tag_sequence + [tag]
                        if self.autoregressive_scheme:
                            if self.autoregressive_scheme == 'teacher_forcing':
                                one_hot_prediction = torch.nn.functional.one_hot(torch.tensor(tag), num_classes=len(self.tagset)).to(torch.float).to(device)
                                one_hot_prediction = one_hot_prediction.unsqueeze(0)
                                new_prev_preds = self.update_previous_predictions(prev_preds.to(device), one_hot_prediction)
                            else:
                                new_prev_preds = self.update_previous_predictions(prev_preds.to(device), scores)
                        else:
                            new_prev_preds = None
                        new_beam.append((combined_score, new_tag_sequence, new_prev_preds))
                new_beam.sort(key=lambda x: x[0], reverse=True)
                beam = new_beam[:beam_size]
            tag_ids = beam[0][1]
        all_tags = list(self.tagset.keys())
        return [all_tags[tag_id] for tag_id in tag_ids]

    def get_scores(self, text, index, previous_predictions, device):
        start = max(0, index - self.window_size // 2)
        end = index + self.window_size // 2 + 1
        window = text[start:end]
        window = [self.vocab.get(char, self.vocab['[UNK]']) for char in window]
        if len(window) < self.window_size:
            pad_left = (self.window_size - len(window)) // 2
            pad_right = self.window_size - len(window) - pad_left
            window = [self.vocab['[PAD]']] * pad_left + window + [self.vocab['[PAD]']] * pad_right
        X = torch.tensor(window).unsqueeze(0).to(device)
        if self.autoregressive_scheme:
            return self(X, extra_logits=previous_predictions.to(device))
        else:
            return self(X)

    def update_previous_predictions(self, previous_predictions, new_prediction):
        if previous_predictions is None:
            return new_prediction
        if previous_predictions.size(0) >= self.tag_context_size:
            previous_predictions = previous_predictions[1:]
        return torch.cat([previous_predictions, new_prediction], dim=0)

    def tag(self, text, device, pos_lm=None, beam_size=None):
        tags = self.decode(text, pos_lm, beam_size, device)
        return [(text[i] if text[i] in self.vocab else '[UNK]', tag) for i, tag in enumerate(tags)]


def train_model(model, model_name, train_loader, validation_loader, criterion, optimizer, num_epochs, device):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X, extra_logits, y in tqdm(train_loader, total=len(train_loader)):
            X = X.to(device)
            if extra_logits is not None:
                extra_logits = extra_logits.to(device)
            y = torch.nn.functional.one_hot(y, num_classes=len(model.tagset)).to(torch.float).to(device)
            optimizer.zero_grad()
            outputs = model(X, extra_logits)
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
            for X_val, extra_logits_val, y_val in validation_loader:
                X_val = X_val.to(device)
                if extra_logits_val is not None:
                    extra_logits_val = extra_logits_val.to(device)
                y_val = torch.nn.functional.one_hot(y_val, num_classes=len(model.tagset)).to(torch.float).to(device)
                outputs_val = model(X_val, extra_logits_val)
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


def load_cc100(split='train'):
    dataset = load_dataset('AlienKevin/cc100-yue-tagged', split=split)
    if split == 'test':
        label_names = dataset.features["pos_tags_ud"].feature.names
        id2label = { i:k for i, k in enumerate(label_names) }

        utterances = []
        for item in dataset:
            utterances.append([(token, id2label[pos]) for token, pos in zip(item['tokens'], item['pos_tags_ud'])])

        return utterances
    else:
        dataset = load_helper(dataset)
        return dataset


def train(model_name, train_loader, validation_loader, vocab, tagset, window_size, tag_context_size, embedding_type, embedding_dim, autoregressive_scheme, network_depth, device):
    model = Tagger(vocab, tagset, window_size, tag_context_size, embedding_type, embedding_dim, autoregressive_scheme, network_depth).to(device)
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


def load_ud_yue(split='test'):
    from datasets import load_dataset

    # Load the universal_dependencies dataset from Hugging Face
    dataset = load_dataset('universal-dependencies/universal_dependencies', 'yue_hk', trust_remote_code=True)

    # Gather all word segmented utterances
    utterances = [[(token, upos_id_to_str(pos)) for token, pos in zip(sentence['tokens'], sentence['upos'])] for sentence in dataset[split]]

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

def test(model_name, test_dataset, pos_lm, beam_size, device):
    from spacy.training import Example
    from spacy.scorer import Scorer
    from spacy.tokens import Doc
    from spacy.vocab import Vocab
    import json

    if test_dataset == 'ud_yue':
        test_dataset = load_ud_yue('test')
    elif test_dataset == 'cc100':
        test_dataset = load_cc100('test')

    random.seed(42)
    random.shuffle(test_dataset)

    test_dataset = test_dataset[:100]

    model = torch.load(f"{model_name}.pth", weights_only=False)
    model.to(device)
    model.eval()

    V = Vocab()
    examples = []
    errors = []
    for reference in tqdm(test_dataset):
        hypothesis = merge_tokens(model.tag(''.join(token for token, _ in reference), device, pos_lm, beam_size))
        reference_tokens = [''.join(char if char in model.vocab.keys() else '[UNK]' for char in token) for token, _ in reference]
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
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help='Mode to run the script in: train or test')
    parser.add_argument('--embedding_type', choices=['one_hot', 'learnable'], required=True, help='Embedding type to use')
    parser.add_argument('--embedding_dim', type=int, default=100, help='Embedding dimension to use')
    parser.add_argument('--vocab_threshold', type=float, default=0.999, help='Vocabulary threshold')
    parser.add_argument('--training_dataset', choices=['hkcancor', 'cc100'], required=True, help='Training dataset to use')
    parser.add_argument('--use_pos_lm', action='store_true', help='Whether to use POS LM during decoding')
    parser.add_argument('--beam_size', type=int, default=None, help='Beam size for beam search')
    parser.add_argument('--window_size', type=int, default=5, help='Window size for the tagger')
    parser.add_argument('--autoregressive_scheme', default=None, choices=['teacher_forcing'])
    parser.add_argument('--tag_context_size', type=int, default=0, help='Tag context size for the tagger')
    parser.add_argument('--network_depth', type=int, default=1, help='Depth of the tagger neural network')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model_name = f"{args.training_dataset}_pos_perceptron{f'_network_depth_{args.network_depth}' if args.network_depth > 1 else ''}_window_{args.window_size}_{args.embedding_type}{f'_{args.autoregressive_scheme}_{args.tag_context_size}' if args.autoregressive_scheme else ''}"

    if args.training_dataset == 'hkcancor':
        training_dataset = load_hkcancor()
    elif args.training_dataset == 'cc100':
        training_dataset = load_cc100()

    random.seed(42)
    random.shuffle(training_dataset)

    validation_dataset = training_dataset[:100]
    train_dataset = training_dataset[100:]

    train_data = TaggerDataset(train_dataset, args.window_size, args.tag_context_size, args.vocab_threshold)
    train_loader = DataLoader(train_data, batch_size=args.batch_size)

    print('Training dataset vocab size:', len(train_data.vocab))
    print('Training dataset tagset size:', len(train_data.tagset))

    validation_data = TaggerDataset(validation_dataset, args.window_size, args.tag_context_size, args.vocab_threshold, vocab=train_data.vocab, tagset=train_data.tagset)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size)

    pos_lm = None
    if args.use_pos_lm:
        pos_lm = LanguageModel([item[1] for item in train_dataset], train_data.tagset, 2)

    if args.mode == 'train':
        train(model_name, train_loader, validation_loader, train_data.vocab, train_data.tagset, args.window_size, args.tag_context_size, args.embedding_type, args.embedding_dim, args.autoregressive_scheme, args.network_depth, device)
    elif args.mode == 'test':
        print('Testing on UD Yue')
        test(model_name, 'ud_yue', pos_lm=pos_lm, beam_size=args.beam_size, device=device)
        print('Testing on CC100')
        test(model_name, 'cc100', pos_lm=pos_lm, beam_size=args.beam_size, device=device)
