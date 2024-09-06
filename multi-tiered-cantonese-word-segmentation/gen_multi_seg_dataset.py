import os
import random
from datasets import Dataset, DatasetDict
import datasets

def get_neighbor_type(char):
    if char == ' ':
        return 'S'
    elif char == '-':
        return 'D'
    elif char == '|':
        return 'P'
    else:
        return 'I'

all_labels = set()

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    chars = []
    labels = []
    for line in lines:
        line = line.strip()
        line_chars = []
        line_labels = []
        for i, char in enumerate(line):
            if char in [' ', '-', '|']:
                continue
            line_chars.append(char)
            left = get_neighbor_type(line[i-1] if i > 0 else ' ')
            right = get_neighbor_type(line[i+1] if i < len(line)-1 else ' ')
            line_labels.append(left + right)
        chars.append(line_chars)
        labels.append(line_labels)
        all_labels.update(line_labels)
    
    return {'chars': chars, 'labels': labels}

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
results = [process_file(os.path.join(data_dir, f)) for f in train_files]
train_data = {'chars': [chars_list for result in results for chars_list in result['chars']], 'labels': [labels_list for result in results for labels_list in result['labels']]}

val_data = process_file(os.path.join(data_dir, val_file))
test_data = process_file(os.path.join(data_dir, test_file))

# Create ClassLabels feature for labels
label_classes = sorted(list(all_labels))

print(label_classes)

label_feature = datasets.Sequence(datasets.ClassLabel(names=label_classes))

# Create datasets and cast labels to ClassLabel
train_dataset = Dataset.from_dict(train_data).cast_column("labels", label_feature)
val_dataset = Dataset.from_dict(val_data).cast_column("labels", label_feature)
test_dataset = Dataset.from_dict(test_data).cast_column("labels", label_feature)

# Combine into a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

# Push to HuggingFace Hub
dataset_dict.push_to_hub("AlienKevin/hkcancor-multi")

print("Dataset uploaded successfully!")
