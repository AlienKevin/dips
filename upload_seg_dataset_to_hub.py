from datasets import Dataset, DatasetDict
import argparse


args = argparse.ArgumentParser()
args.add_argument('--dataset', type=str, choices=['as', 'cityu', 'msr', 'pku'], required=True)
args = args.parse_args()


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip().split() for line in f]

# Read the data
train_data = read_file(f'data/sighan2005/{args.dataset}_training.utf8')
test_data = read_file(f'data/sighan2005/{args.dataset}_test_gold.utf8')

# Create datasets
train_dataset = Dataset.from_dict({"tokens": train_data})
test_dataset = Dataset.from_dict({"tokens": test_data})

# Split train dataset into train and validation
train_val = train_dataset.train_test_split(test_size=0.1, seed=42)

# Create the final dataset dictionary
dataset_dict = DatasetDict({
    "train": train_val["train"],
    "validation": train_val["test"],
    "test": test_dataset
})

# Print dataset info
print(dataset_dict)

# Upload to Hugging Face Hub
dataset_dict.push_to_hub(f"AlienKevin/{args.dataset}-seg")
