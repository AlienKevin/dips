from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
import datasets
import json
import argparse


args = argparse.ArgumentParser()
args.add_argument('--dataset', type=str, choices=['hkcancor', 'cc100_yue', 'lihkg', 'wiki_yue_long', 'genius'], required=True)
args = args.parse_args()


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

all_pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

if args.dataset == 'hkcancor':
    # Load HKCanCor dataset
    dataset = load_dataset('nanyang-technological-university-singapore/hkcancor')

    label_names = dataset["train"].features["pos_tags_ud"].feature.names
    id2label = { i:k for i, k in enumerate(label_names) }

    def process_hkcancor(example):
        # Map tag ids to labels
        pos_tags_ud = [id2label[tag] for tag in example['pos_tags_ud']]

        # Remap 'V' tag to 'VERB' in pos_tags_ud
        pos_tags_ud = ['VERB' if tag == 'V' else tag for tag in pos_tags_ud]
        
        return {
            'tokens': example['tokens'],
            'pos_tags_prf': example['pos_tags_prf'],
            'pos_tags_ud': pos_tags_ud,
            'sentence': ''.join(example['tokens']),
            'sentence_preserved': True,
            'conversation_id': example['conversation_id'],
            'speaker': example['speaker'],
            'turn_number': example['turn_number'],
            'transcriptions': example['transcriptions']
        }
    
    dataset = dataset.map(process_hkcancor)
    
    # Define features
    features = datasets.Features({
        "tokens": datasets.Sequence(datasets.Value("string")),
        "pos_tags_prf": datasets.Sequence(datasets.Value("string")),
        "pos_tags_ud": datasets.Sequence(datasets.features.ClassLabel(names=all_pos_tags)),
        "sentence": datasets.Value("string"),
        "sentence_preserved": datasets.Value("bool"),
        "conversation_id": datasets.Value("string"),
        "speaker": datasets.Value("string"),
        "turn_number": datasets.Value("int16"),
        "transcriptions": datasets.Sequence(datasets.Value("string")),
    })
    
    # Convert to DatasetDict with the correct features and create train/validation/test splits
    full_dataset = Dataset.from_dict(dataset['train'].to_dict(), features=features)
    
    # Shuffle the dataset
    full_dataset = full_dataset.shuffle(seed=42)
    
    # Split the dataset
    train_testvalid = full_dataset.train_test_split(test_size=0.1 * 2, shuffle=False)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, shuffle=False)
    
    dataset_dict = DatasetDict({
        'train': train_testvalid['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    })
    
    # Upload to Hugging Face Hub
    dataset_dict.push_to_hub("AlienKevin/hkcancor")
    exit(0)


# Existing code for other datasets
pos_results = read_jsonl(f'{args.dataset}_outputs_v2/pos_results.jsonl')
pos_errors = read_jsonl(f'{args.dataset}_outputs_v2/pos_errors.jsonl')

for entry in pos_results:
    entry['sentence_preserved'] = True

# Only keep error entries if the error is "Segmentation result does not match the input sentence"
# and for those entries, parse the "result" field as a json and get its pos_tagged_words field
filtered_errors = []
for entry in pos_errors:
    try:
        result = json.loads(entry['result'])
        pos_tagged_words = result.get('pos_tagged_words')
        if isinstance(pos_tagged_words, list) and all(isinstance(word, str) and isinstance(tag, str) for word, tag in pos_tagged_words):
            if all(tag in all_pos_tags for _, tag in pos_tagged_words) and all(len(word) > 0 for word, _ in pos_tagged_words):
                entry['result'] = pos_tagged_words
                del entry['error']
                entry['sentence_preserved'] = False
                filtered_errors.append(entry)
    except Exception as e:
        continue

pos_errors = filtered_errors

combined_data = pos_results + pos_errors

# Filter out entries with inputs containing specific substrings
if args.dataset == 'cc100_yue':
    raw_length = len(combined_data)
    filtered_data = [entry for entry in combined_data if '嘅發音' not in entry['input'] and 'Hotels.com' not in entry['input']]
    print(f"Number of low-quality entries filtered: {raw_length - len(filtered_data)}")

filtered_data = [entry for entry in combined_data if len(entry['input']) > 0 and len(entry['result']) > 0]

# Map the result field to tokens and pos_tags_ud
for entry in filtered_data:
    tokens, pos_tags = zip(*entry['result'])
    entry['tokens'] = list(tokens)
    entry['pos_tags_ud'] = list(pos_tags)
    del entry['result']
    entry['sentence'] = entry['input']
    del entry['input']

# Define the pos_tags_ud feature using the reference dataset
pos_tags_ud = datasets.Sequence(
    datasets.features.ClassLabel(names=all_pos_tags)
)

# Create a Hugging Face dataset
features = datasets.Features(
    {
        "sentence": datasets.Value("string"),
        "tokens": datasets.Sequence(datasets.Value("string")),
        "pos_tags_ud": pos_tags_ud,
        "sentence_preserved": datasets.Value("bool"),
    }
)
filtered_data_dict = {key: [entry[key] for entry in filtered_data] for key in filtered_data[0]}
dataset = Dataset.from_dict(filtered_data_dict, features=features)

# Calculate the percentage of sentence_preserved entries
sentence_preserved_ratio = sum(dataset['sentence_preserved']) / len(dataset)

# Calculate the test and validation sizes based on the total dataset
test_size = 0.10
validation_size = 0.10

# Adjust the split ratios for sentence_preserved entries
sentence_preserved_test_size = test_size / sentence_preserved_ratio
sentence_preserved_validation_size = validation_size / sentence_preserved_ratio

# Split the dataset based on sentence_preserved
sentence_preserved = dataset.filter(lambda example: example['sentence_preserved'])
not_sentence_preserved = dataset.filter(lambda example: not example['sentence_preserved'])

# Split sentence_preserved data into train, validation, and test
sentence_preserved_splits = sentence_preserved.train_test_split(test_size=sentence_preserved_test_size)
sentence_preserved_train_val = sentence_preserved_splits['train'].train_test_split(test_size=sentence_preserved_validation_size / (1 - sentence_preserved_test_size))

# Combine the splits
dataset_dict = DatasetDict({
    # Only use sentence_preserved entries for testing and validation
    "train": concatenate_datasets([sentence_preserved_train_val['train'], not_sentence_preserved]),
    "validation": sentence_preserved_train_val['test'],
    "test": sentence_preserved_splits['test']
})

# Shuffle the training set
dataset_dict['train'] = dataset_dict['train'].shuffle(seed=42)

print(dataset_dict)

# Visualize the first 3 entries
import pprint

pp = pprint.PrettyPrinter()
pp.pprint(list(dataset_dict["train"].select(range(1))))

# Upload to Hugging Face Hub
if args.dataset == 'cc100_yue':
    dataset_dict.push_to_hub("AlienKevin/cc100-yue-tagged")
elif args.dataset == 'lihkg':
    dataset_dict.push_to_hub("AlienKevin/lihkg-tagged")
elif args.dataset == 'wiki_yue_long':
    dataset_dict.push_to_hub("AlienKevin/wiki-yue-long-tagged")
elif args.dataset == 'genius':
    dataset_dict.push_to_hub("AlienKevin/genius-tagged")
