import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from tagged_graph_dataset import TaggedGraphDataset
from tqdm import tqdm
import argparse
from tagger_dataset import load_tagged_dataset
import datasets
import time
import os
import wandb


class GCNTagger(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GCNTagger, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        x = self.output(x)
        return x


def validate_model(model, dataset, device, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            out = model(batch['x'].to(device), batch['edge_index'].to(device))
            loss = F.cross_entropy(out, batch['y'].to(device))
            total_loss += loss.item()

            predictions = out.argmax(dim=-1)
            correct = (predictions == batch['y'].to(device))
            total_correct += correct.sum().item()
            total_tokens += batch['y'].numel()

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0

    return avg_loss, accuracy


def train_model(model_path, train_dataset, validation_dataset, device, num_epochs=10, batch_size=256, lr=2e-05, hidden_dim=256, num_layers=2, validation_steps=0.2):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = GCNTagger(input_dim=train_dataset.embedding_dim, 
                      hidden_dim=hidden_dim, 
                      output_dim=len(train_dataset.tagged_dataset.features['tags'].feature.names), 
                      num_layers=num_layers).to(device)

    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # Initialize wandb
    wandb.init(project="graph-tagger", name=os.path.splitext(os.path.basename(model_path))[0])

    # Training loop
    best_val_loss = float('inf')
    global_step = 0
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            out = model(batch['x'].to(device), batch['edge_index'].to(device))
            loss = F.cross_entropy(out, batch['y'].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            wandb.log({"train_loss": loss.item()}, step=global_step)

            if global_step % round(validation_steps * len(train_loader)) == 0:
                val_loss, val_accuracy = validate_model(model, validation_dataset, device, batch_size)
                wandb.log({
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy
                }, step=global_step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model, model_path)

                model.train()  # Set the model back to training mode
            
            global_step += 1

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch, "avg_train_loss": avg_loss}, step=global_step)
        scheduler.step()

    wandb.finish()
    return model


def load_lexicon(word_embedding_path):
    for line in open(word_embedding_path, 'r', encoding='utf-8'):
        word, embedding = line.strip().split(' ', 1)
        yield word


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test the POS tagger model.')
    parser.add_argument('--training_dataset', nargs='+', choices=['hkcancor', 'cc100-yue', 'lihkg', 'wiki-yue-long', 'genius', 'ctb8', 'msr-seg', 'as-seg', 'cityu-seg', 'pku-seg'], required=True, help='Training dataset(s) to use')
    parser.add_argument('--word_embedding_path', type=str, required=True, help='Path to the word embedding file')
    parser.add_argument('--char_embedding_path', type=str, required=True, help='Path to the character embedding file')
    parser.add_argument('--tagging_scheme', type=str, choices=['BI', 'BIES'], default='BIES', help='Tagging scheme to use')
    args = parser.parse_args()

    model_name = f'gcn_tagger_{'_'.join(args.training_dataset)}_{time.strftime("%m%d-%H%M")}'
    model_path = f'models/{model_name}.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    train_datasets = [load_tagged_dataset(dataset, 'train', args.tagging_scheme) for dataset in args.training_dataset]
    validation_datasets = [load_tagged_dataset(dataset, 'validation', args.tagging_scheme) for dataset in args.training_dataset]

    train_dataset = datasets.concatenate_datasets(train_datasets)
    validation_dataset = datasets.concatenate_datasets(validation_datasets)

    train_dataset = TaggedGraphDataset(tagged_dataset=train_dataset, lexicon=list(load_lexicon(args.word_embedding_path)), char_embedding_path=args.char_embedding_path)
    validation_dataset = TaggedGraphDataset(tagged_dataset=validation_dataset, lexicon=list(load_lexicon(args.word_embedding_path)), char_embedding_path=args.char_embedding_path, vocab=train_dataset.vocab)

    model = train_model(model_path, train_dataset, validation_dataset, device)
