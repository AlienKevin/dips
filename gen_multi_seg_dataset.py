from datasets import load_dataset, Dataset, DatasetDict
from transformers import TokenClassificationPipeline, AutoTokenizer, AutoModelForTokenClassification
import torch
import datasets
from tqdm import tqdm

batch_size = 64

# Load the dataset
# dataset = load_dataset("raptorkwok/cantonese_sentences")
# dataset['train'] = dataset['train'].select(range(200000))
# field_name = 'content'

dataset = load_dataset("R5dwMg/zh-wiki-yue-long")
field_name = 'text'

# Split the dataset into train and val_test
train_val_test = dataset['train'].train_test_split(test_size=min(20000, int(len(dataset['train']) * 0.1)), seed=42)
train = train_val_test['train']
val_test = train_val_test['test']

# Split val_test into validation and test
val_test_split = val_test.train_test_split(test_size=0.5, seed=42)
validation = val_test_split['train']
test = val_test_split['test']

# Create a new DatasetDict with the splits
dataset = DatasetDict({
    'train': train,
    'validation': validation,
    'test': test
})

print(f"Train size: {len(dataset['train'])}")
print(f"Validation size: {len(dataset['validation'])}")
print(f"Test size: {len(dataset['test'])}")


# Load the ELECTRA model for token classification
model_name = "finetune-ckip-transformers/electra_base_hkcancor_multi"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)


class LogitsTokenClassificationPipeline(TokenClassificationPipeline):
    def postprocess(self, model_outputs):
        assert(len(model_outputs) == 1)
        
        input_ids = model_outputs[0]["input_ids"][0]
        special_tokens_mask = model_outputs[0]["special_tokens_mask"][0]
        logits = model_outputs[0]["logits"][0]
        
        input_ids = input_ids[special_tokens_mask == 0]
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        logits = logits[special_tokens_mask == 0]
        
        return input_tokens, logits
        


nlp = LogitsTokenClassificationPipeline(model=model, device="mps" if torch.backends.mps.is_available() else "cpu", tokenizer=tokenizer)

def process_batch(batch):
    texts = [text[:510] for text in batch[field_name]]  # Truncate to 512 characters
    outputs = nlp(texts, batch_size=batch_size)
    
    processed_batch = []
    for tokens, token_logits in outputs:
        chars = []
        logits = []
        labels = []
        for token, logit in zip(tokens, token_logits):
            token = token.removeprefix("##")
            chars.append(token[0])
            logits.append(logit.tolist())
            labels.append(logit.argmax().item())
            for char in token[1:]:
                chars.append(char)
                logits.append([0, 1, 0, 0]) # I
                labels.append(1) # I
        processed_batch.append({'chars': chars, 'labels': labels, 'logits': logits})
    
    return processed_batch

# Process the dataset
processed_data = {'train': [], 'validation': [], 'test': []}

for split in ['train', 'validation', 'test']:
    for i in tqdm(range(0, len(dataset[split]), batch_size)):  # Process in batches
        batch = dataset[split][i:i+batch_size]
        results = process_batch(batch)
        processed_data[split].extend(results)

# Create datasets
features=datasets.Features({
    'chars': datasets.Sequence(datasets.Value('string')),
    'labels': datasets.Sequence(datasets.ClassLabel(names=['D', 'I', 'P', 'S'])),
    'logits': datasets.Sequence(datasets.Sequence(datasets.Value('float32'), length=4))
})
train_dataset = Dataset.from_list(processed_data['train'], features=features)
val_dataset = Dataset.from_list(processed_data['validation'], features=features)
test_dataset = Dataset.from_list(processed_data['test'], features=features)

# Combine into a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

dataset_dict = dataset_dict.filter(lambda example: len(example['chars']) > 0)

# Push to HuggingFace Hub
dataset_dict.push_to_hub("AlienKevin/lihkg-multi")

print("Dataset uploaded successfully!")
