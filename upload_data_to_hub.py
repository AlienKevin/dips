from datasets import Dataset, DatasetDict
import datasets
import json


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# Read the data from the JSONL files
pos_results = read_jsonl('outputs_v2/pos_results.jsonl')
pos_errors = read_jsonl('outputs_v2/pos_errors.jsonl')

for entry in pos_results:
    entry['sentence_preserved'] = True

all_pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

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
raw_length = len(combined_data)
filtered_data = [entry for entry in combined_data if '嘅發音' not in entry['input'] and 'Hotels.com' not in entry['input']]
print(f"Number of low-quality entries filtered: {raw_length - len(filtered_data)}")

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
split_dataset = dataset.train_test_split(test_size=0.05)
dataset_dict = DatasetDict({"train": split_dataset["train"], "test": split_dataset["test"]})

print(dataset_dict)

# Visualize the first 3 entries
import pprint

pp = pprint.PrettyPrinter()
pp.pprint(list(dataset_dict["train"].select(range(1))))


# Upload to Hugging Face Hub
dataset_dict.push_to_hub("AlienKevin/cc100-yue-tagged")
