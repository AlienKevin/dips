import os
import random
from datasets import Dataset, DatasetDict
from datasets import load_dataset
import re

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    tokens = []
    for line in lines:
        # Split by spaces, hyphens, and vertical bars
        line_tokens = re.split(r'[ -|]', line.strip())
        # Remove empty tokens
        line_tokens = [token for token in line_tokens if token]
        tokens.append(line_tokens)
    
    return {'tokens': tokens}

# Get all txt files in the data directory
data_dir = 'data'
txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]

# Randomly select files for validation and test
random.seed(42)
random.shuffle(txt_files)
val_file = txt_files.pop()
test_file = txt_files.pop()
train_files = txt_files

# Process files
train_data = {'tokens': [tokens_list for f in train_files for tokens_list in process_file(os.path.join(data_dir, f))['tokens']]}
val_data = process_file(os.path.join(data_dir, val_file))
test_data = process_file(os.path.join(data_dir, test_file))

# Create datasets
train_dataset = Dataset.from_dict(train_data)
val_dataset = Dataset.from_dict(val_data)
test_dataset = Dataset.from_dict(test_data)

# Combine into a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

# Push to HuggingFace Hub
dataset_dict.push_to_hub("AlienKevin/hkcancor-fine")

print("Dataset uploaded successfully!")
