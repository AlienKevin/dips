import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from collections import Counter
import random
from tagger_dataset import TaggerDataset, load_tagged_dataset
from utils import normalize, pad_batch_seq, merge_tokens, score_tags, read_pretrained_embeddings
from tqdm import tqdm
import json
from vocab import Vocab
import wandb
import os
import time


# Load BPE mappings
def load_bpe_mappings(file_path):
    bpe_mappings = {}
    code_to_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            char, code = line.strip().split('\t')
            codes = code.split(' ')
            bpe_mappings[char] = codes
            for code in codes:
                if code not in code_to_index:
                    code_to_index[code] = len(code_to_index)

    # Map each individual code to the unicode codepoint at (0x4E00 + code_to_index[code]) in bpe_mappings
    # i.e. replace cangjie code with the corresponding Chinese character
    for char, codes in bpe_mappings.items():
        bpe_mappings[char] = [chr(0x4E00 + code_to_index[code]) for code in codes]

    return bpe_mappings


bpe_mappings = load_bpe_mappings('data/Cangjie5_SC_BPE.txt')


class BertConfig:
    def __init__(
        self,
        hidden_size: int = 200,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 2,
        intermediate_size: int = 200*2,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 20,
        initializer_range: float = 0.0,
        **kwargs,  # unused
    ):
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers

    @staticmethod
    def from_json(path: str) -> "BertConfig":
        with open(path, "r") as f:
            file_content = json.load(f)

        return BertConfig(**file_content)


class ConvConfig:
    def __init__(self, embedding_size=200, hidden_sizes=[200, 200, 200, 200, 200], kernel_size=3):
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.kernel_size = kernel_size
    
    @staticmethod
    def from_json(path: str) -> "ConvConfig":
        with open(path, "r") as f:
            file_content = json.load(f)

        return ConvConfig(**file_content)


class Segmenter(nn.Module):
    def __init__(self, cangjie_expand, vocab, config=ConvConfig(), tagset=None, char_embedding_path=None):
        super(Segmenter, self).__init__()
        self.vocab = vocab

        self.config = config

        if isinstance(config, ConvConfig):
            embedding_size = config.embedding_size
            output_size = config.hidden_sizes[-1]
        elif isinstance(config, BertConfig):
            embedding_size = config.hidden_size
            output_size = config.hidden_size
        
        if char_embedding_path is None:
            self.embedding = nn.Embedding(len(vocab), embedding_size)
        else:
            self.embedding = read_pretrained_embeddings(char_embedding_path, vocab, freeze=False)
            print('Loaded embeddings')
        self.tagset = tagset
        self.cangjie_expand = cangjie_expand

        if isinstance(config, ConvConfig):
            self.conv_layers = nn.ModuleList([
                nn.Conv1d(
                    in_channels=embedding_size if i == 0 else config.hidden_sizes[i-1],
                    out_channels=config.hidden_sizes[i],
                    kernel_size=config.kernel_size,
                    dilation=2**i if self.cangjie_expand else 1,
                    padding='same'
                ) for i in range(len(config.hidden_sizes))
            ])
            self.relu = nn.ReLU()
            self.output = nn.Linear(output_size, len(vocab) if not tagset else len(tagset))
        elif isinstance(config, BertConfig):
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

            self.embedding_layer_norm = nn.LayerNorm(config.hidden_size)
            self.embedding_dropout = nn.Dropout(p=config.hidden_dropout_prob)

            self.encoders = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.hidden_dropout_prob,
                    activation=config.hidden_act,
                    batch_first=True,
                ),
                num_layers=config.num_hidden_layers,
            )
            self.output = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, len(vocab) if not tagset else len(tagset), bias=True)
            )

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = self.embedding(x)

        if isinstance(self.config, ConvConfig):
            x = x.transpose(1, 2)
            # x shape: (batch_size, embedding_dim, sequence_length)
            for conv in self.conv_layers:
                x = self.relu(conv(x))
            # x shape: (batch_size, hidden_dim, sequence_length)
            x = x.transpose(1, 2)
            # x shape: (batch_size, sequence_length, hidden_dim)
        elif isinstance(self.config, BertConfig):
            seq_length = x.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
            x = x + self.position_embeddings(position_ids).unsqueeze(0)
            x = self.embedding_layer_norm(x)
            x = self.embedding_dropout(x)
            x = self.encoders(x)

        return self.output(x)
    
    def save(self, path):
        if self.tagset:
            torch.save({
                'state_dict': self.state_dict(),
                'vocab': self.vocab.token2id_map,
                'tagset': self.tagset,
                'config': self.config,
                'cangjie_expand': self.cangjie_expand
            }, path)
        else:
            torch.save({
                'state_dict': self.state_dict(),
                'vocab': self.vocab.token2id_map,
                'config': self.config,
                'cangjie_expand': self.cangjie_expand
            }, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, weights_only=False)
        config = checkpoint.get('config')
        vocab = Vocab(checkpoint['vocab'])
        tagset = checkpoint.get('tagset')
        cangjie_expand = checkpoint.get('cangjie_expand', False)
        model = cls(cangjie_expand, vocab, config=config, tagset=tagset)
        model.load_state_dict(checkpoint['state_dict'])
        return model
    
    def tag(self, text):
        # Step 1: Normalize text and map Chinese characters to Cangjie codes
        normalized_text = normalize(text)
        mapped_text = []
        token_map = {}
        for i, char in enumerate(normalized_text):
            if self.cangjie_expand and char in bpe_mappings:
                cangjie_codes = bpe_mappings[char]
                mapped_text.extend(cangjie_codes)
            else:
                mapped_text.append(char)
            token_map[len(mapped_text) - 1] = i

        # Step 2: Convert characters to input_ids
        input_ids = [self.vocab[char] for char in mapped_text]
        
        # Step 3: Run the model and get top tag predictions
        tagged_text = []
        if isinstance(self.config, BertConfig):
            # Windowed inference to overcome context size limit of BERT
            # Reference: https://hanlp.hankcs.com/docs/_modules/hanlp/transform/transformer_tokenizer.html
            max_length = self.config.max_position_embeddings
            chunk_size = max_length
            overlap = max_length // 2
            
            for start in range(0, len(input_ids), chunk_size - overlap):
                end = min(start + chunk_size, len(input_ids))
                chunk_input_ids = input_ids[start:end]
                
                with torch.no_grad():
                    input_tensor = torch.tensor(chunk_input_ids).unsqueeze(0)  # Add batch dimension
                    output = self(input_tensor)
                    predictions = torch.argmax(output, dim=-1).squeeze(0)
                
                # Map predicted tags back to the original text for this chunk
                for i, pred in enumerate(predictions):
                    global_index = start + i
                    if global_index in token_map and (start == 0 or i >= overlap):
                        char = text[token_map[global_index]]
                        tag = self.tagset.id2token(pred.item())
                        tagged_text.append((char, tag))
                
                if end == len(input_ids):
                    break
        else:
            with torch.no_grad():
                input_tensor = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension
                output = self(input_tensor)
                predictions = torch.argmax(output, dim=-1).squeeze(0)

            # Map predicted tags back to the original text
            for i, pred in enumerate(predictions):
                if i in token_map:
                    char = text[token_map[i]]
                    tag = self.tagset.id2token(pred.item())
                    tagged_text.append((char, tag))

        return tagged_text


class MLMIterableDataset(IterableDataset):
    def __init__(self, cangjie_expand, dataset, field_name, vocab_threshold, vocab=None, mask_prob=0.15):
        self.cangjie_expand = cangjie_expand
        self.dataset = dataset
        self.field_name = field_name
        self.vocab = vocab if vocab else self.build_vocabulary(vocab_threshold)
        self.mask_prob = mask_prob

    def build_vocabulary(self, vocab_threshold):
        def count_tokens(item):
            counter = Counter()
            sentence = normalize(item[self.field_name])
            for char in sentence:
                if self.cangjie_expand and char in bpe_mappings:
                    counter.update(bpe_mappings[char])
                else:
                    counter[char] += 1
            # Have to serialize to string because pyarrow doesn't support serialization of Counter
            return {"counter": json.dumps(dict(counter))}
        
        counters = self.dataset.select(range(100000)).map(count_tokens, batched=False, num_proc=10)
        counter = Counter()
        for c in counters['counter']:
            counter.update(json.loads(c))

        total_count = sum(counter.values())
        vocab = {'[PAD]': 0, '[UNK]': 1, '[MASK]': 2}
        current_count = 0

        for token, count in counter.most_common():
            vocab[token] = len(vocab)
            current_count += count
            if current_count / total_count > vocab_threshold:
                break

        return Vocab(vocab)
    
    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for item in self.dataset:
            sentence = item[self.field_name]
            # Normalize to half-width
            sentence = normalize(sentence)
            # Expand to BPE
            tokens = []
            for char in sentence:
                if self.cangjie_expand and char in bpe_mappings:
                    tokens.extend(bpe_mappings[char])
                else:
                    tokens.append(char)
            # Convert to vocab indices
            input_ids = [self.vocab[token] for token in tokens]
            # Apply masking
            masked_input_ids = input_ids.copy()
            i = 0
            while i < len(masked_input_ids):
                if random.random() < self.mask_prob:
                    mask_length = 1
                    for j in range(i, i + mask_length):
                        masked_input_ids[j] = self.vocab['[MASK]']
                    i += mask_length
                else:
                    i += 1
            yield torch.tensor(masked_input_ids, dtype=torch.int64), torch.tensor(input_ids, dtype=torch.int64)


def validate(model, validation_dataset, validation_dataloader, criterion, mlm_loss, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for input_ids, labels in validation_dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids)
            
            if mlm_loss:
                # Calculate loss only on masked tokens
                mask = input_ids == model.vocab['[MASK]']
                loss = criterion(outputs[mask], labels[mask])
            else:
                mask = labels != -100
                # Don't need to apply mask here because criterion already ignores padding tokens
                loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))

            total_loss += loss.item()

            # Calculate accuracy
            predictions = outputs.argmax(dim=-1)
            correct = (predictions == labels) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / len(validation_dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0

    tag_measures = None
    if not mlm_loss:
        model.to('cpu')
        tag_measures, tag_errors = score_tags(validation_dataset, lambda text: model.tag(text))
        model.to(device)

    return avg_loss, accuracy, tag_measures



def train(model, model_path, train_dataloader, validation_dataset, validation_dataloader, optimizer, scheduler, criterion, device, mlm_loss=True, num_epochs=40, validation_steps=0.2):
    model.train()
    best_val_loss = float('inf')
    global_step = 0

    wandb.init(project="conv-mlm", name=os.path.splitext(os.path.basename(model_path))[0])

    for epoch in range(num_epochs):
        for batch, (input_ids, labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            optimizer.zero_grad()
            outputs = model(input_ids.to(device))
            if mlm_loss:
                labels[input_ids != model.vocab['[MASK]']] = model.vocab['[PAD]'] # only calculate loss on masked tokens
                loss = criterion(outputs.view(-1, len(model.vocab)), labels.view(-1).to(device))
            else:
                loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1).to(device))
            loss.backward()
            optimizer.step()

            wandb.log({"train_loss": loss.item()}, step=global_step)

            if global_step % round(validation_steps * len(train_dataloader)) == 0:
                val_loss, val_accuracy, tag_measures = validate(model, validation_dataset, validation_dataloader, criterion, mlm_loss, device)
                if tag_measures is None:
                    wandb.log({
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy
                    }, step=global_step)
                else:
                    wandb.log({
                        "val_loss": val_loss,
                        "val_acc": val_accuracy,
                        "val_pos_acc": tag_measures['pos_acc'],
                        "val_token_f1": tag_measures['token_f'],
                        "val_token_p": tag_measures['token_p'],
                        "val_token_r": tag_measures['token_r']
                    }, step=global_step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save(model_path)

                model.train()  # Set the model back to training mode
            
            global_step += 1
            scheduler.step()


def cangjie_expand(token, tag):
    results = []
    for char in token:
        if char in bpe_mappings:
            codes = bpe_mappings[char]
            results.extend([(code, '[PAD]') for code in codes[:-1]] + [(codes[-1], tag)])
        else:
            results.append((char, tag))
    return results


def load_train_dataset(args):
    if args.train_dataset == 'rthk':
        dataset_author = 'jed351'
        dataset_name = 'rthk_news'
        field_name = 'content'
    elif args.train_dataset == 'genius':
        dataset_author = 'beyond'
        dataset_name = 'chinese_clean_passages_80m'
        field_name = 'passage'
    elif args.train_dataset == 'tte':
        dataset_author = 'liswei'
        dataset_name = 'Taiwan-Text-Excellence-2B'
        field_name = 'text'
    elif args.train_dataset.endswith('-seg') or args.train_dataset == 'ctb8':
        dataset_author = 'AlienKevin'
        dataset_name = args.train_dataset
    else:
        raise ValueError("Invalid dataset choice")

    if args.train_dataset in ['rthk', 'tte', 'genius', 'tte']:
        dataset = load_dataset(f'{dataset_author}/{dataset_name}', split='train')

        # Split the dataset into train and validation sets
        train_test_split = dataset.train_test_split(test_size=min(0.05*len(dataset), 10000))
        train_dataset = train_test_split['train']
        validation_dataset = train_test_split['test']

        # Create dataset and dataloader
        train_dataset = MLMIterableDataset(args.cangjie_expand, train_dataset, field_name, args.vocab_threshold)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      collate_fn=lambda batch: pad_batch_seq(batch, max_sequence_length=args.max_sequence_length))

        validation_dataset = MLMIterableDataset(args.cangjie_expand, validation_dataset, field_name, args.vocab_threshold, vocab=train_dataset.vocab)
        validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size,
                                           collate_fn=lambda batch: pad_batch_seq(batch, max_sequence_length=args.max_sequence_length))
    else:
        train_dataset = load_tagged_dataset(dataset_name, split='train', tagging_scheme=args.tagging_scheme,
                                            transform=cangjie_expand if args.cangjie_expand else None)
        validation_dataset = load_tagged_dataset(dataset_name, split='validation', tagging_scheme=args.tagging_scheme,
                                                transform=cangjie_expand if args.cangjie_expand else None)

        train_dataset = TaggerDataset(train_dataset, window_size=-1, tag_context_size=-1, vocab_threshold=args.vocab_threshold, sliding=False)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      collate_fn=lambda batch: pad_batch_seq(batch, max_sequence_length=args.max_sequence_length))

        print('Training dataset vocab size:', len(train_dataset.vocab))
        print('Training dataset tagset size:', len(train_dataset.tagset))

        validation_dataset = TaggerDataset(validation_dataset, window_size=-1, tag_context_size=-1, vocab_threshold=args.vocab_threshold, vocab=train_dataset.vocab, tagset=train_dataset.tagset, sliding=False)
        validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size,
                                           collate_fn=lambda batch: pad_batch_seq(batch, max_sequence_length=args.max_sequence_length))
    
        # Reload validation in the same format as the test dataset for calculating F1 score
        validation_dataset = load_tagged_dataset(dataset_name, split='validation', tagging_scheme=args.tagging_scheme, output_format='test')

    return train_dataset, train_dataloader, validation_dataset, validation_dataloader


def train_model(args, model_path, device):
    train_dataset, train_dataloader, validation_dataset, validation_dataloader = load_train_dataset(args)

    tagset = train_dataset.tagset if hasattr(train_dataset, 'tagset') else None
    model = Segmenter(args.cangjie_expand, train_dataset.vocab, config=args.config, tagset=tagset, char_embedding_path=args.char_embedding_path)
    model.to(device)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters())
    total_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(0.1 * len(train_dataloader))  # 10% of first epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=total_steps,
                                                    pct_start=warmup_steps/total_steps,
                                                    anneal_strategy='linear', div_factor=25.0,
                                                    final_div_factor=10000.0)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train(model, model_path, train_dataloader, validation_dataset, validation_dataloader, optimizer, scheduler, criterion, device=device, mlm_loss=not args.segmentation, num_epochs=args.num_epochs, validation_steps=args.validation_steps)


def test_model(args, model_path):
    test_dataset = load_tagged_dataset(args.test_dataset, 'test')
    
    model = Segmenter.load(model_path)

    results, errors = score_tags(test_dataset, lambda text: model.tag(text))

    if not args.segmentation:
        print(f"POS Tagging Accuracy: {results['pos_acc']}")
        import json
        # Write errors to errors.jsonl
        with open('errors.jsonl', 'w', encoding='utf-8') as f:
            for error in errors:
                json.dump(error, f, ensure_ascii=False)
                f.write('\n')
    else:
        import json
        # Write errors to errors.jsonl
        with open('errors.jsonl', 'w', encoding='utf-8') as f:
            for error in errors:
                json.dump({"REF": ' '.join(token for token, _ in error['reference']), "HYP": ' '.join(token for token, _ in error['hypothesis'])}, f, ensure_ascii=False)
                f.write('\n')
    
    print(f"Errors have been written to errors.jsonl")
    
    print(f"Token F1 Score: {results['token_f']}")
    print(f"Token Precision: {results['token_p']}")
    print(f"Token Recall: {results['token_r']}")


def infer_model(args, model_path):
    model = Segmenter.load(model_path)

    for text in open(args.test_file, 'r', encoding='utf-8').readlines():
        tagged = merge_tokens(model.tag(text))
        if args.segmentation:
            print(' '.join(token for token, _ in tagged))
        else:
            print(tagged)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    import argparse

    parser = argparse.ArgumentParser(description='Train segmenter model on selected dataset')
    parser.add_argument('--mode', type=str, choices=['train', 'infer', 'test'], required=True, help='Mode to run in')
    parser.add_argument('--model_path', type=str, help='Path to model')
    parser.add_argument('--config', type=str, choices=['conv', 'bert'], default='conv', help='Architecture to use')
    parser.add_argument('--train_dataset', type=str, choices=['rthk', 'genius', 'tte', 'cityu-seg', 'as-seg', 'msr-seg', 'pku-seg', 'genius-seg', 'ctb8'],
                        help='Dataset to use for training')
    parser.add_argument('--test_dataset', type=str, choices=['as-seg', 'cityu-seg', 'msr-seg', 'pku-seg', 'genius-seg', 'ctb8'],
                        help='Dataset to use for testing')
    parser.add_argument('--test_file', type=str, help='File to use for testing')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train for')
    parser.add_argument('--validation_steps', type=float, default=0.2, help='Validation steps')
    parser.add_argument('--max_sequence_length', type=int, default=20, help='Maximum sequence length')
    parser.add_argument('--segmentation', action='store_true', help='Do segmentation rather than MLM')
    parser.add_argument('--tagging_scheme', type=str, choices=['BI', 'BIES'], default='BIES', help='Tagging scheme for segmentation')
    parser.add_argument('--vocab_threshold', type=float, default=0.9999, help='Vocabulary threshold')
    parser.add_argument('--cangjie_expand', action='store_true', help='Expand characters into their Cangjie codes')
    parser.add_argument('--char_embedding_path', type=str, help='Path to character embeddings')
    args = parser.parse_args()

    if args.model_path:
        model_path = args.model_path
    else:
        model_path = f'models/{args.config}_{args.train_dataset}{'_mlm' if not args.segmentation else ''}_{time.strftime("%m%d-%H%M")}.pth'

    args.config = ConvConfig() if args.config == 'conv' else BertConfig()

    if args.mode == 'train':
        train_model(args, model_path, device)
    elif args.mode == 'infer':
        infer_model(args, model_path)
    elif args.mode == 'test':
        test_model(args, model_path)

if __name__ == "__main__":
    main()
