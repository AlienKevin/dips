import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
import datasets
import random
from tqdm import tqdm
import math
import wandb
from crf import CRF


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
        if self.tag_context_size > 0 or not self.sliding:
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
                X = torch.tensor([self.vocab.get(char, self.vocab['[UNK]']) for char in sentence])
                y = torch.tensor([self.tagset[tag] for tag in tags])
                yield (X, y)


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


def read_pretrained_embeddings(embedding_path, vocab):
    word_to_embed = {}
    unknown_embeds = []
    with open(embedding_path, "r", encoding="utf-8") as f:
        for line in f:
            split = line.split(' ')
            if len(split) > 2:
                word = split[0]
                vec = torch.tensor([float(x) for x in split[1:]])
                if word in vocab:
                    word_to_embed[word] = vec
                else:
                    unknown_embeds.append(vec)
    
    embedding_dim = next(iter(word_to_embed.values())).size(0)
    out = torch.empty(len(vocab), embedding_dim)
    nn.init.uniform_(out, -0.8, 0.8)
    
    for word, embed in word_to_embed.items():
        out[vocab[word]] = embed
    
    if unknown_embeds:
        unk_embed = torch.stack(unknown_embeds).mean(dim=0)
        out[vocab['[UNK]']] = unk_embed
    
    return nn.Embedding.from_pretrained(out, freeze=True)


def create_positional_encoding(d_model, max_len):
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    return pe


class Tagger(nn.Module):
    def __init__(self, vocab, tagset, embedding_dim, embedding_path, network_type, network_depth, kernel_sizes):
        super(Tagger, self).__init__()
        self.vocab = vocab
        self.tagset = tagset
        if embedding_path is None:
            self.embedding = nn.Embedding(len(vocab), embedding_dim)
        else:
            self.embedding = read_pretrained_embeddings(embedding_path, vocab)

        self.network_type = network_type
        self.network_depth = network_depth

        # feature_dims = [128, 256, 512]
        feature_dims = [64, 128, 256]

        if 'cnn' in network_type:
            self.conv1 = nn.ModuleList([
                nn.Conv1d(embedding_dim, feature_dims[0], kernel_size=k, padding='same')
                for k in kernel_sizes
            ])

            if network_depth >= 2:
                self.conv2 = nn.Conv1d(feature_dims[0] * len(kernel_sizes), feature_dims[1], kernel_size=3, padding='same', dilation=2 if 'dilated' in network_type else 1)
            if network_depth >= 3:
                self.conv3 = nn.Conv1d(feature_dims[1], feature_dims[2], kernel_size=3, padding='same', dilation=2 if 'dilated' in network_type else 1)
            
            self.fc = nn.Linear(feature_dims[network_depth - 1], len(tagset))
        elif 'bi-lstm' in network_type:
            self.lstm = nn.LSTM(embedding_dim * (2 if 'bigram' in network_type else 1), 100, num_layers=network_depth, bidirectional=True, batch_first=True, dropout=0.2)
            # Double the feature dimension because it's bidirectional
            self.fc = nn.Linear(100 * 2, len(tagset))
        elif network_type == 'mha':
            encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=2)
            self.mha = nn.TransformerEncoder(encoder_layer, num_layers=network_depth)
            self.fc = nn.Linear(embedding_dim, len(tagset))
            self.positional_encoding = create_positional_encoding(d_model=embedding_dim, max_len=1000)
        
        self.use_crf = 'crf' in network_type
        if self.use_crf:
            self.crf = CRF(len(tagset), batch_first=True)
        
    def forward(self, x, tags=None):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        
        if 'bigram' in self.network_type:
            # Pad the end of embedded with an extra [PAD] token
            pad_token = self.embedding(torch.tensor([self.vocab['[PAD]']]).to(embedded.device))
            padded_embedded = torch.cat([embedded, pad_token.expand(embedded.size(0), 1, -1)], dim=1)
            bigrams = torch.cat([padded_embedded[:, :-1], padded_embedded[:, 1:]], dim=2)
            # Take the average of the bigram embeddings
            bigrams = (bigrams[:, :, :embedded.size(2)] + bigrams[:, :, embedded.size(2):]) / 2
            embedded = torch.cat([embedded, bigrams], dim=2)
        
        if 'cnn' in self.network_type:
            # Transpose for 1D convolution
            x = embedded.transpose(1, 2)  # (batch_size, embedding_dim, sequence_length)
            
            conv_outputs = []
            for conv in self.conv1:
                conv_output = F.relu(conv(x))
                conv_outputs.append(conv_output)
            x = torch.cat(conv_outputs, dim=1)
            if self.network_depth >= 2:
                x = F.relu(self.conv2(x))
            if self.network_depth >= 3:
                x = F.relu(self.conv3(x))
            
            # Transpose back
            x = x.transpose(1, 2)  # (batch_size, sequence_length, 512)
        elif 'bi-lstm' in self.network_type:
            x, _ = self.lstm(embedded)
        elif self.network_type == 'mha':
            # Transpose x to (seq_len, batch_size, embedding_dim)
            x = embedded.transpose(0, 1)
            x = x + self.positional_encoding[:x.size(0)].to(x.device)
            x = self.mha(x)
            x = x.transpose(0, 1)

        # Apply fully connected layer to each time step
        emissions = self.fc(x)  # (batch_size, sequence_length, len(tagset))
        
        if self.use_crf:
            if tags is not None:
                # Training mode
                return -self.crf(emissions, tags, reduction="token_mean")  # Negative log-likelihood
            else:
                # Inference mode
                return self.crf.decode(emissions)
        else:
            return emissions

    def tag(self, text, device):
        # Convert text to tensor of indices
        indices = torch.tensor([self.vocab.get(char, self.vocab['[UNK]']) for char in text], dtype=torch.long).unsqueeze(0).to(device)
        
        # Get model predictions
        with torch.no_grad():
            if self.use_crf:
                predicted_tags = self(indices)[0]  # CRF returns a list of lists, we take the first one
            else:
                outputs = self(indices)
                _, predicted_tags = torch.max(outputs, dim=2)
                predicted_tags = predicted_tags[0]  # Remove batch dimension
        
        # Convert tag indices to tag strings
        all_tags = list(self.tagset.keys())
        tags = [all_tags[tag_id] for tag_id in predicted_tags]
        
        # Pair each character with its predicted tag
        return [(char, tag) for char, tag in zip(text, tags)]


class SlidingTagger(nn.Module):
    def __init__(self, vocab, tagset, window_size, tag_context_size, embedding_type, embedding_dim, autoregressive_scheme, network_type, network_depth):
        super(SlidingTagger, self).__init__()
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
        
        self.embedding_dim = embedding_dim

        self.network_type = network_type
        if network_type == 'mlp':
            if network_depth == 3:
                self.linear = nn.Sequential(
                    nn.Linear(embedding_dim * window_size + (tag_context_size * len(self.tagset) if self.autoregressive_scheme else 0), embedding_dim),
                    nn.ReLU(),
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.ReLU(),
                    nn.Linear(embedding_dim, len(tagset))
                )
            elif network_depth == 2:
                self.linear = nn.Sequential(
                    nn.Linear(embedding_dim * window_size + (tag_context_size * len(self.tagset) if self.autoregressive_scheme else 0), embedding_dim),
                    nn.ReLU(),
                    nn.Linear(embedding_dim, len(tagset))
                )
            elif network_depth == 1:
                self.linear = nn.Linear(embedding_dim * window_size + (tag_context_size * len(self.tagset) if self.autoregressive_scheme else 0), len(tagset))
            else:
                raise ValueError(f"Invalid network depth: {network_depth}")
        elif network_type == 'cnn':
            full_embedding_dim = embedding_dim + (len(self.tagset) if self.autoregressive_scheme else 0)

            if network_depth == 2:
                layers = [
                    nn.Conv1d(full_embedding_dim, full_embedding_dim * 2, kernel_size=2, stride=1),
                    nn.ReLU(),
                    nn.Conv1d(full_embedding_dim * 2, full_embedding_dim * 4, kernel_size=2, stride=1),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1),
                    nn.Flatten(),
                    nn.Linear(full_embedding_dim * 4, len(tagset))
                ]

            self.cnn = nn.Sequential(*layers)
        elif network_type == 'mha':
            num_heads = 8
            
            self.mha_layers = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=num_heads) for _ in range(network_depth)])
            self.positional_encoding = create_positional_encoding(d_model=self.embedding_dim, max_len=window_size)
            
            self.fc = nn.Linear(self.embedding_dim, len(tagset))
        else:
            raise ValueError(f"Invalid network type: {network_type}")

    def get_embedding(self, char):
        if char not in self.vocab:
            char = '[UNK]'
        return self.embedding(torch.tensor(self.vocab[char]).to(device))

    def forward(self, x, extra_logits=None):
        x = self.embedding(x)
        if self.network_type == 'mlp':
            x = x.view(x.size(0), -1)  # Flatten the input
            if extra_logits is not None:
                extra_logits = extra_logits.view(extra_logits.size(0), -1)
                x = torch.cat([x, extra_logits], dim=1)
            return self.linear(x)
        elif self.network_type == 'cnn':
            if extra_logits is not None:
                batch_size_x, seq_len, embed_dim = x.shape
                batch_size_extra, extra_len, extra_dim = extra_logits.shape

                assert(batch_size_x == batch_size_extra)
                assert(extra_len <= seq_len)
                
                # Pad extra_logits to match the sequence length of x
                padded_extra_logits = torch.nn.functional.pad(extra_logits, (0, 0, 0, seq_len - extra_len))
                
                # Concatenate x and padded_extra_logits along the last dimension
                x = torch.cat([x, padded_extra_logits], dim=-1)

                assert(x.shape == (batch_size_x, seq_len, embed_dim + extra_dim))
                
                x = x.transpose(1, 2)

                assert(x.shape == (batch_size_x, embed_dim + extra_dim, seq_len))
            return self.cnn(x)
        elif self.network_type == 'mha':
            # Transpose x to (seq_len, batch_size, embedding_dim)
            x = x.transpose(0, 1)
            
            # Add positional encoding
            x = x + self.positional_encoding[:x.size(0), :].to(x.device)
            
            # Apply multi-head attention layers
            attn_output = x
            for mha_layer in self.mha_layers:
                attn_output, _ = mha_layer(attn_output, attn_output, attn_output)
            
            # Transpose back to (batch_size, seq_len, embedding_dim)
            attn_output = attn_output.transpose(0, 1)
            
            # Get the middle character's output
            middle_index = attn_output.size(1) // 2
            middle_output = attn_output[:, middle_index, :]
            
            # Apply final linear layer to the middle character's output
            return self.fc(middle_output)

    def decode(self, text, pos_lm, beam_size, device):
        if self.autoregressive_scheme:
            pad_tag_index = self.tagset['[PAD]']
            pad_one_hot = torch.nn.functional.one_hot(torch.tensor(pad_tag_index), num_classes=len(self.tagset)).float()
            previous_predictions = torch.stack([pad_one_hot] * self.tag_context_size)
        else:
            previous_predictions = None
        if pos_lm is None:
            # Greedy decoding
            tag_ids = []
            for i in range(len(text)):
                scores = self.get_scores(text, i, device, previous_predictions)
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
                    scores = self.get_scores(text, i, device, prev_preds)
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

    def get_scores(self, text, index, device, previous_predictions=None):
        start = max(0, index - self.window_size // 2)
        end = index + self.window_size // 2 + 1
        window = text[start:end]
        window = [self.vocab.get(char, self.vocab['[UNK]']) for char in window]
        if len(window) < self.window_size:
            pad_left = (self.window_size - len(window)) // 2
            pad_right = self.window_size - len(window) - pad_left
            window = [self.vocab['[PAD]']] * pad_left + window + [self.vocab['[PAD]']] * pad_right
        # Add batch dimension
        X = torch.tensor(window).unsqueeze(0).to(device)
        if self.autoregressive_scheme:
            previous_predictions = previous_predictions.unsqueeze(0).to(device)
            return self(X, extra_logits=previous_predictions)
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
        return [(text[i], tag) for i, tag in enumerate(tags)]


def train_model(model, model_name, train_loader, validation_loader, optimizer, scheduler, num_epochs, device, training_log_steps=10, validation_steps=0.1):
    best_loss = float('inf')
    step = 0

    wandb.init(project="cantag", name=model_name)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X, y in tqdm(train_loader, total=len(train_loader)):
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            if model.use_crf:
                loss = model(X, y)  # The forward method now returns the negative log-likelihood for CRF
            else:
                outputs = model(X)
                loss = F.cross_entropy(outputs.view(-1, outputs.shape[-1]), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            step += 1
            if step % training_log_steps == 0:
                avg_loss = total_loss / training_log_steps
                wandb.log({"train_loss": avg_loss}, step=step)
                total_loss = 0
            
            if step % round(validation_steps * len(train_loader)) == 0:
                model.eval()
                validation_loss = 0
                with torch.no_grad():
                    for X_val, y_val in validation_loader:
                        X_val = X_val.to(device)
                        y_val = y_val.to(device)
                        if model.use_crf:
                            loss_val = model(X_val, y_val)
                        else:
                            outputs_val = model(X_val)
                            loss_val = F.cross_entropy(outputs_val.view(-1, outputs_val.shape[-1]), y_val.view(-1))
                        validation_loss += loss_val.item()
                avg_validation_loss = validation_loss / len(validation_loader)
                wandb.log({"validation_loss": avg_validation_loss}, step=step)
                print(f"Step {step}, Validation Loss: {avg_validation_loss}")
                
                if avg_validation_loss < best_loss:
                    best_loss = avg_validation_loss
                    torch.save(model, f"models/{model_name}.pth")
                
                model.train()
        
        scheduler.step()


def train_sliding_model(model, model_name, train_loader, validation_loader, optimizer, scheduler, num_epochs, device, training_log_steps=10, validation_steps=0.1):
    best_loss = float('inf')
    step = 0
    criterion = nn.CrossEntropyLoss()

    wandb.init(project="cantag", name=model_name)

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
            
            step += 1
            if step % training_log_steps == 0:
                avg_loss = total_loss / training_log_steps
                wandb.log({"train_loss": avg_loss}, step=step)
                total_loss = 0
            
            if step % round(validation_steps * len(train_loader)) == 0:
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
                wandb.log({"validation_loss": avg_validation_loss}, step=step)
                print(f"Step {step}, Validation Loss: {avg_validation_loss}")
                
                if avg_validation_loss < best_loss:
                    best_loss = avg_validation_loss
                    torch.save(model, f"models/{model_name}.pth")
                
                model.train()
        
        scheduler.step()

        print(f"Epoch {epoch + 1}/{num_epochs} completed")


def load_helper(dataset, tagging_scheme):
    label_names = dataset.features["pos_tags_ud"].feature.names
    id2label = { i:k for i, k in enumerate(label_names) }

    tagged_corpus = []
    for item in dataset:
        tokens = item['tokens']
        tags = [id2label[tag] for tag in item['pos_tags_ud']]
        char_tokens = []
        char_tags = []
        for token, tag in zip(tokens, tags):
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
    from datasets import load_dataset

    # Load the universal_dependencies dataset from Hugging Face
    dataset = load_dataset('universal-dependencies/universal_dependencies', lang, trust_remote_code=True)
    
    dataset = dataset.map(lambda example: {
        'tokens': [normalize(token) for token in example['tokens']]
    })

    # Gather all word segmented utterances
    utterances = [[(token, upos_id_to_str(pos)) for token, pos in zip(sentence['tokens'], sentence['upos'])] for sentence in dataset[split]]

    return utterances


def load_tagged_dataset(dataset_name, split, tagging_scheme=None):
    dataset = load_dataset(f'AlienKevin/{(f"{dataset_name}-tagged") if dataset_name not in ["ctb8", "msr-seg", "as-seg", "cityu-seg", "pku-seg"] else dataset_name}' , split=split)

    if dataset_name.endswith('-seg'):
        dataset = dataset.map(lambda example: {
            'tokens': example['tokens'],
            'pos_tags_ud': ['X' for _ in example['tokens']]
        }, features=datasets.Features({
            'tokens': datasets.Sequence(datasets.features.Value('string')),
            'pos_tags_ud': datasets.Sequence(datasets.features.ClassLabel(names=['X']))
        }))

    dataset = dataset.map(lambda example: {
        'tokens': [normalize(token) for token in example['tokens']]
    })
    if split == 'test':
        label_names = dataset.features["pos_tags_ud"].feature.names
        id2label = { i:k for i, k in enumerate(label_names) }

        utterances = []
        for item in dataset:
            utterances.append([(token, id2label[pos]) for token, pos in zip(item['tokens'], item['pos_tags_ud'])])

        return utterances
    else:
        dataset = load_helper(dataset, tagging_scheme)
        return dataset


def train(model_name, model, train_loader, validation_loader, num_epochs, learning_rate, learning_rate_decay, sliding, device):
    torch.save(model, f"models/{model_name}.pth")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=learning_rate_decay)

    if sliding:
        train_sliding_model(model, model_name, train_loader, validation_loader, optimizer, scheduler, num_epochs=num_epochs, device=device)
    else:
        train_model(model, model_name, train_loader, validation_loader, optimizer, scheduler, num_epochs=num_epochs, device=device)


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


def merge_tokens(tagged_characters):
    merged_tokens = []
    current_token = []
    current_tag = None

    for char, tag in tagged_characters:
        if tag.startswith('B-') or tag.startswith('S-'):
            if current_token:
                merged_tokens.append((''.join(current_token), current_tag))
            current_token = [char]
            current_tag = tag[2:]
        elif tag.startswith('I-') or tag.startswith('E-'):
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


def fix_tag(tag):
    if tag == 'V':
        return 'VERB'
    elif tag == '[PAD]':
        return 'NOUN'
    return tag


def infer(model, text, sliding, pos_lm, beam_size, device):
    model.eval()

    text = normalize(text)

    if sliding:
        hypothesis = merge_tokens(model.tag(text, device, pos_lm, beam_size))
    else:
        hypothesis = merge_tokens(model.tag(text, device))

    return hypothesis


def test(model, test_dataset, sliding, pos_lm, beam_size, segmentation_only, device):
    from spacy.training import Example
    from spacy.scorer import Scorer
    from spacy.tokens import Doc
    from spacy.vocab import Vocab
    import json

    if test_dataset.startswith('ud_'):
        test_dataset = load_ud(test_dataset.removeprefix('ud_'), 'test')
    else:
        test_dataset = load_tagged_dataset(test_dataset, 'test')

    if segmentation_only:
        test_dataset = [[(token, 'X') for token, _ in utterance] for utterance in test_dataset]

    random.seed(42)
    random.shuffle(test_dataset)

    test_dataset = test_dataset[:100]

    model.eval()

    V = Vocab()
    examples = []
    errors = []
    for reference in tqdm(test_dataset):
        if sliding:
            hypothesis = merge_tokens(model.tag(''.join(token for token, _ in reference), device, pos_lm, beam_size))
        else:
            hypothesis = merge_tokens(model.tag(''.join(token for token, _ in reference), device))
        reference_tokens = [token for token, _ in reference]
        target = Doc(V, words=reference_tokens, spaces=[False for _ in reference], pos=[fix_tag(tag) for _, tag in reference])
        predicted_doc = Doc(V, words=[token for token, _ in hypothesis], spaces=[False for _ in hypothesis], pos=[fix_tag(tag) for _, tag in hypothesis])
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


def load_model(model_name, vocab, tagset, sliding, window_size, tag_context_size, embedding_type, embedding_dim, embedding_path, autoregressive_scheme, network_type, network_depth, kernel_sizes, device, load_weights=False):
    if sliding:
        model = SlidingTagger(vocab, tagset, window_size, tag_context_size, embedding_type, embedding_dim, autoregressive_scheme, network_type, network_depth).to(device)
    else:
        model = Tagger(vocab, tagset, embedding_dim, embedding_path, network_type, network_depth, kernel_sizes).to(device)
    
    if load_weights:
        model.load_state_dict(torch.load(f"models/{model_name}.pth", weights_only=False).state_dict())

    return model.to(device)


# Read the STCharacters.txt file and create a mapping dictionary
t2s = {}
with open('data/STCharacters.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            simplified, traditional = parts
            for char in traditional.split():
                t2s[char] = simplified


def normalize(text):
    """
    Simplify traditional Chinese text to simplified Chinese.

    Args:
        text (str): The input traditional Chinese text.

    Returns:
        str: The simplified Chinese text.

    Example:
        >>> normalize("漢字")
        '汉字'
        >>> normalize("這是一個測試")
        '这是一个测试'
        >>> normalize("Hello, 世界!")
        'Hello, 世界!'
    """
    return ''.join(t2s.get(char, char) for char in text)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train or test the POS tagger model.')
    parser.add_argument('--mode', choices=['train', 'test', 'infer'], required=True, help='Mode to run the script in: train or test')
    parser.add_argument('--text', type=str, default=None, help='Text to infer')
    parser.add_argument('--embedding_type', choices=['one_hot', 'learnable'], help='Embedding type to use')
    parser.add_argument('--embedding_dim', type=int, default=100, help='Embedding dimension to use')
    parser.add_argument('--embedding_path', default=None, help='Path to the character embedding file')
    parser.add_argument('--vocab_threshold', type=float, default=0.9999, help='Vocabulary threshold')
    parser.add_argument('--training_dataset', nargs='+', choices=['hkcancor', 'cc100-yue', 'lihkg', 'wiki-yue-long', 'genius', 'ctb8', 'msr-seg', 'as-seg', 'cityu-seg', 'pku-seg'], required=True, help='Training dataset(s) to use')
    parser.add_argument('--sliding', action='store_true', help='Whether to use sliding window')
    parser.add_argument('--tagging_scheme', choices=['BI', 'BIES'], default='BI', help='Tagging scheme to use')
    parser.add_argument('--use_pos_lm', action='store_true', help='Whether to use POS LM during decoding')
    parser.add_argument('--beam_size', type=int, default=None, help='Beam size for beam search')
    parser.add_argument('--window_size', type=int, default=5, help='Window size for the tagger')
    parser.add_argument('--autoregressive_scheme', default=None, choices=['teacher_forcing'])
    parser.add_argument('--tag_context_size', type=int, default=0, help='Tag context size for the tagger')
    parser.add_argument('--network_depth', type=int, default=1, help='Depth of the tagger neural network')
    parser.add_argument('--network_type', choices=['mlp', 'cnn', 'mha', 'dilated_cnn', 'cnn_crf', 'dilated_cnn_crf', 'bi-lstm', 'bi-lstm-bigram'], default='mlp', help='Type of the tagger neural network')
    parser.add_argument('--kernel_sizes', nargs='+', type=int, default=[3], help='Kernel sizes for the CNN')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--learning_rate_decay', type=float, default=0.0, help='Learning rate decay for the optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--segmentation_only', action='store_true', help='Whether to only segment the text')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model_name = f"pos_tagger_{'_'.join(args.training_dataset)}{f'_sliding' if args.sliding else ''}{f'_seg' if args.segmentation_only else ''}{f'_{args.tagging_scheme}' if args.tagging_scheme != 'BI' else ''}{f'_window_size_{args.window_size}' if (args.window_size != 5 and args.sliding) else ''}{f'_{args.embedding_type}' if args.embedding_type else ''}{f'_embedding_dim_{args.embedding_dim}' if args.embedding_dim != 100 else ''}{f'_{args.network_type}' if args.network_type != 'mlp' else ''}{f'_network_depth_{args.network_depth}' if args.network_depth > 1 else ''}{f'_{'_'.join(map(str, args.kernel_sizes))}' if args.kernel_sizes != [3] else ''}{f'_{args.autoregressive_scheme}_{args.tag_context_size}' if args.autoregressive_scheme else ''}"

    train_dataset = []
    validation_dataset = []
    for dataset in args.training_dataset:
        train_dataset.extend(load_tagged_dataset(dataset, 'train', args.tagging_scheme))
        validation_dataset.extend(load_tagged_dataset(dataset, 'validation', args.tagging_scheme))

    random.seed(42)
    random.shuffle(train_dataset)

    if args.segmentation_only:
        train_dataset = [(tokens, [tag[:2] + 'X' for tag in tags]) for (tokens, tags) in train_dataset]
        validation_dataset = [(tokens, [tag[:2] + 'X' for tag in tags]) for (tokens, tags) in validation_dataset]

    def sliding_collate_fn(batch):
        X, extra_logits, y = zip(*batch)
        X = torch.stack(X)
        y = torch.stack(y)
        if extra_logits[0] is not None:
            extra_logits = torch.stack(extra_logits)
        else:
            extra_logits = None
        return X, extra_logits, y

    def collate_fn(batch):
        X = [item[0] for item in batch]
        y = [item[1] for item in batch]
        X_padded = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=train_data.vocab['[PAD]'])
        y_padded = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=train_data.tagset['[PAD]'])
        return X_padded, y_padded

    train_data = TaggerDataset(train_dataset, args.window_size, args.tag_context_size, args.vocab_threshold, sliding=args.sliding)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=collate_fn if not args.sliding else sliding_collate_fn)

    print('Training dataset vocab size:', len(train_data.vocab))
    print('Training dataset tagset size:', len(train_data.tagset))

    validation_data = TaggerDataset(validation_dataset, args.window_size, args.tag_context_size, args.vocab_threshold, vocab=train_data.vocab, tagset=train_data.tagset, sliding=args.sliding)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, collate_fn=collate_fn if not args.sliding else sliding_collate_fn)

    pos_lm = None
    if args.use_pos_lm:
        pos_lm = LanguageModel([item[1] for item in train_dataset], train_data.tagset, 2)

    model = load_model(model_name, train_data.vocab, train_data.tagset, sliding=args.sliding,
                       window_size=args.window_size, tag_context_size=args.tag_context_size,
                       embedding_type=args.embedding_type, embedding_dim=args.embedding_dim, embedding_path=args.embedding_path,
                       autoregressive_scheme=args.autoregressive_scheme,
                       network_type=args.network_type, network_depth=args.network_depth,
                       kernel_sizes=args.kernel_sizes, device=device,
                       load_weights=args.mode in ['test', 'infer'])

    if args.mode == 'train':
        train(model_name, model, train_loader, validation_loader, args.num_epochs, args.learning_rate, args.learning_rate_decay, args.sliding, device)
    elif args.mode == 'test':
        print('Testing on UD Yue')
        test(model, 'ud_yue_hk', sliding=args.sliding, pos_lm=pos_lm, beam_size=args.beam_size, segmentation_only=args.segmentation_only, device=device)
        print('Testing on UD ZH-HK')
        test(model, 'ud_zh_hk', sliding=args.sliding, pos_lm=pos_lm, beam_size=args.beam_size, segmentation_only=args.segmentation_only, device=device)
        print('Testing on CTB 8.0')
        test(model, 'ctb8', sliding=args.sliding, pos_lm=pos_lm, beam_size=args.beam_size, segmentation_only=args.segmentation_only, device=device)
        if args.segmentation_only:
            print('Testing on MSR')
            test(model, 'msr-seg', sliding=args.sliding, pos_lm=pos_lm, beam_size=args.beam_size, segmentation_only=args.segmentation_only, device=device)
            print('Testing on CityU')
            test(model, 'cityu-seg', sliding=args.sliding, pos_lm=pos_lm, beam_size=args.beam_size, segmentation_only=args.segmentation_only, device=device)
            print('Testing on AS')
            test(model, 'as-seg', sliding=args.sliding, pos_lm=pos_lm, beam_size=args.beam_size, segmentation_only=args.segmentation_only, device=device)
            print('Testing on PKU')
            test(model, 'pku-seg', sliding=args.sliding, pos_lm=pos_lm, beam_size=args.beam_size, segmentation_only=args.segmentation_only, device=device)
        print('Testing on LIHKG')
        test(model, 'lihkg', sliding=args.sliding, pos_lm=pos_lm, beam_size=args.beam_size, segmentation_only=args.segmentation_only, device=device)
        print('Testing on CC100')
        test(model, 'cc100-yue', sliding=args.sliding, pos_lm=pos_lm, beam_size=args.beam_size, segmentation_only=args.segmentation_only, device=device)
    elif args.mode == 'infer':
        hypothesis = infer(model, args.text, sliding=args.sliding, pos_lm=pos_lm, beam_size=args.beam_size, device=device)
        formatted_hypothesis = ' '.join([f"{token}/{tag}" for token, tag in hypothesis])
        print(f"{formatted_hypothesis}")
        exit(0)
