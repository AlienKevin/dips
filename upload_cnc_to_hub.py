from datasets import Dataset
from datasets import Features, Sequence, Value
import datasets
from collections import defaultdict

def parse_cnc_line(line):
    tokens = []
    pos_tags_cnc = []
    for item in line.strip().split():
        if '/' in item:
            token, tag = item.rsplit('/', 1)
            tokens.append(token)
            pos_tags_cnc.append(tag)
    return tokens, pos_tags_cnc

def read_cnc_file(file_path):
    data = defaultdict(list)
    all_tags = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens, pos_tags_cnc = parse_cnc_line(line)
            if tokens:
                data['tokens'].append(tokens)
                data['pos_tags_cnc'].append(pos_tags_cnc)
                data['sentence'].append(''.join(tokens))
                data['sentence_preserved'].append(True)
                all_tags.update(pos_tags_cnc)

    all_tags_sorted = sorted(list(all_tags))
    
    return data, all_tags_sorted
# Read and parse the CNC files
train_file_path = 'data/cnc/train.txt'
test_file_path = 'data/cnc/test.txt'
dev_file_path = 'data/cnc/dev.txt'

train_data, train_pos_tags = read_cnc_file(train_file_path)
test_data, test_pos_tags = read_cnc_file(test_file_path)
dev_data, dev_pos_tags = read_cnc_file(dev_file_path)

all_pos_tags = sorted(list(set(train_pos_tags + test_pos_tags + dev_pos_tags)))

print(all_pos_tags)

assert(len(all_pos_tags) == 33)

features = Features({
    "tokens": Sequence(Value("string")),
    "pos_tags_cnc": Sequence(datasets.features.ClassLabel(names=all_pos_tags)),
    "sentence": Value("string"),
    "sentence_preserved": Value("bool"),
})

# Create Dataset objects for each split
train_dataset = Dataset.from_dict(train_data, features=features)
test_dataset = Dataset.from_dict(test_data, features=features)
validation_dataset = Dataset.from_dict(dev_data, features=features)

# Combine datasets into a DatasetDict
dataset_dict = datasets.DatasetDict({
    "train": train_dataset,
    "test": test_dataset,
    "validation": validation_dataset
})

print(dataset_dict["train"][0])

# Print some information about the datasets
print(f"Train dataset size: {len(dataset_dict['train'])}")
print(f"Test dataset size: {len(dataset_dict['test'])}")
print(f"Validation dataset size: {len(dataset_dict['validation'])}")
print(f"Number of unique POS tags: {len(all_pos_tags)}")
print(f"All POS tags: {all_pos_tags}")

# Upload to Hugging Face Hub
dataset_dict.push_to_hub("AlienKevin/cnc")
