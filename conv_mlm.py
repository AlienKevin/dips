import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from collections import Counter
import unicodedata
import random
from utils import normalize, pad_batch_seq
from tqdm import tqdm
import json
import wandb

class ConvMLM(nn.Module):
    def __init__(self, vocab, embedding_dim=100, hidden_dim=100, num_layers=4):
        super(ConvMLM, self).__init__()
        self.vocab = vocab
        self.embedding = nn.Embedding(len(vocab), embedding_dim)
        self.vocab_inv = {v: k for k, v in vocab.items()}
        
        # Dilated convolutions
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim if i == 0 else hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                dilation=2**i,
                padding='same'
            ) for i in range(num_layers)
        ])
        
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_dim, len(vocab))

    def get_token(self, id):
        return self.vocab_inv.get(id)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = self.embedding(x).transpose(1, 2)
        # x shape: (batch_size, embedding_dim, sequence_length)

        for conv in self.conv_layers:
            x = self.relu(conv(x))

        # x shape: (batch_size, hidden_dim, sequence_length)
        x = x.transpose(1, 2)
        # x shape: (batch_size, sequence_length, hidden_dim)

        return self.output(x)

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

    # Map each individual code to f'[{code_index}]' in bpe_mappings
    for char, codes in bpe_mappings.items():
        bpe_mappings[char] = [f'[{code_to_index[code]}]' for code in codes]

    return bpe_mappings

class MLMIterableDataset(IterableDataset):
    def __init__(self, dataset_name, split, mask_prob=0.05):
        self.dataset = load_dataset(dataset_name, split=split)
        self.bpe_mappings = load_bpe_mappings('data/Cangjie5_SC_BPE.txt')
        self.vocab = self.build_vocabulary()
        self.mask_prob = mask_prob

    def build_vocabulary(self):
        def count_tokens(item):
            counter = Counter()
            sentence = normalize(unicodedata.normalize('NFKC', item['content']))
            for char in sentence:
                if char in self.bpe_mappings:
                    counter.update(self.bpe_mappings[char])
                else:
                    counter[char] += 1
            # Have to serialize to string because pyarrow doesn't support serialization of Counter
            return {"counter": json.dumps(dict(counter))}
        
        counters = self.dataset.map(count_tokens, batched=False, num_proc=10)
        counter = Counter()
        for c in counters['counter']:
            counter.update(json.loads(c))

        total_count = sum(counter.values())
        vocab = {'[PAD]': 0, '[UNK]': 1, '[MASK]': 2}
        current_count = 0

        for token, count in counter.most_common():
            vocab[token] = len(vocab)
            current_count += count
            if current_count / total_count > 0.9999:
                break

        return vocab
    
    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for item in self.dataset:
            sentence = item['content']
            # Normalize to half-width
            sentence = normalize(unicodedata.normalize('NFKC', sentence))
            # Expand to BPE
            tokens = []
            for char in sentence:
                if char in self.bpe_mappings:
                    tokens.extend(self.bpe_mappings[char])
                else:
                    tokens.append(char)
            # Convert to vocab indices
            input_ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
            # Apply masking
            masked_input_ids = input_ids.copy()
            i = 0
            while i < len(masked_input_ids):
                if random.random() < self.mask_prob:
                    mask_length = int(max(1, min(10, len(masked_input_ids) - i, round(random.gauss(6, 2)))))
                    for j in range(i, i + mask_length):
                        masked_input_ids[j] = self.vocab['[MASK]']
                    i += mask_length
                else:
                    i += 1
            yield torch.tensor(masked_input_ids, dtype=torch.int64), torch.tensor(input_ids, dtype=torch.int64)


def validate(model, validation_dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for input_ids, labels in validation_dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids)
            
            # Calculate loss only on masked tokens
            mask = input_ids == model.vocab['[MASK]']
            loss = criterion(outputs[mask], labels[mask])
            total_loss += loss.item()

            # Calculate accuracy
            predictions = outputs.argmax(dim=-1)
            correct = (predictions == labels) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / len(validation_dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0

    return avg_loss, accuracy



def train(model, dataset_name, train_dataloader, validation_dataloader, optimizer, scheduler, criterion, device, num_epochs=40, validation_steps=0.2):
    model.train()
    best_val_loss = float('inf')
    global_step = 0

    wandb.init(project="conv-mlm", name=dataset_name)

    for epoch in range(num_epochs):
        for batch, (input_ids, labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            optimizer.zero_grad()
            outputs = model(input_ids.to(device))
            labels[input_ids != model.vocab['[MASK]']] = model.vocab['[PAD]'] # only calculate loss on masked tokens
            loss = criterion(outputs.view(-1, len(model.vocab)), labels.view(-1).to(device))
            loss.backward()
            optimizer.step()

            wandb.log({"train_loss": loss.item()}, step=global_step)

            if global_step % round(validation_steps * len(train_dataloader)) == 0:
                val_loss, val_accuracy = validate(model, validation_dataloader, criterion, device)
                wandb.log({
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy
                }, step=global_step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), 'models/conv_mlm.pth')

                model.train()  # Set the model back to training mode
            
            global_step += 1
        scheduler.step()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    dataset_author = 'jed351'
    dataset_name = 'rthk_news'

    # Create dataset and dataloader
    train_dataset = MLMIterableDataset(f'{dataset_author}/{dataset_name}', 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=256, collate_fn=lambda batch: pad_batch_seq(batch, train_dataset.vocab['[PAD]']))

    validation_dataset = MLMIterableDataset(f'{dataset_author}/{dataset_name}', 'validation')
    validation_dataloader = DataLoader(validation_dataset, batch_size=256, collate_fn=lambda batch: pad_batch_seq(batch, validation_dataset.vocab['[PAD]']))

    model = ConvMLM(train_dataset.vocab)
    model.to(device)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab['[PAD]'])

    # Train the model
    train(model, dataset_name, train_dataloader, validation_dataloader, optimizer, scheduler, criterion, device=device)

if __name__ == "__main__":
    main()
