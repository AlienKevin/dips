from datasets import load_dataset
import random

# Load the LIHKG dataset
dataset = load_dataset("AlienKevin/cc100-yue-tagged")

label_names = dataset["test"].features["pos_tags_ud"].feature.names
id2label = { i:k for i, k in enumerate(label_names) }

# Function to format tokens and tags
def format_tokens_tags(tokens, tags):
    return ' '.join([f"{token}/{id2label[tag]}" for token, tag in zip(tokens, tags)])

# Get a random sample from the dataset
sample_size = 10
random_samples = random.sample(list(dataset['train']), sample_size)

# Print the formatted tokens and tags for each sample
for i, sample in enumerate(random_samples, 1):
    tokens = sample['tokens']
    tags = sample['pos_tags_ud']
    formatted = format_tokens_tags(tokens, tags)
    print(f"Sample {i}:")
    print(formatted)
    print()
