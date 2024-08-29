from datasets import load_dataset, concatenate_datasets, DatasetDict
import hanlp

def merge_short_sentences(sentences):
    # Merge extremely short sentences
    merged_sentences = []
    i = 0
    while i < len(sentences):
        current_sentence = sentences[i]
        if len(current_sentence) <= 10:
            if i > 0 and i < len(sentences) - 1:
                # If not the first or last sentence, merge with shorter neighbor
                prev_sentence = merged_sentences[-1]
                next_sentence = sentences[i + 1]
                if len(prev_sentence) <= len(next_sentence):
                    merged_sentences[-1] = prev_sentence + current_sentence
                else:
                    merged_sentences.append(current_sentence + next_sentence)
                    i += 1  # Skip next sentence as it's been merged
            elif i == 0 and len(sentences) > 1:
                # If first sentence, merge with next
                merged_sentences.append(current_sentence + sentences[i + 1])
                i += 1  # Skip next sentence as it's been merged
            elif i == len(sentences) - 1 and len(merged_sentences) > 0:
                # If last sentence, merge with previous
                merged_sentences[-1] = merged_sentences[-1] + current_sentence
            else:
                # If only sentence in group, keep as is
                merged_sentences.append(current_sentence)
        else:
            merged_sentences.append(current_sentence)
        i += 1
    return merged_sentences

def split_into_sentences(examples, field, source):
    split_sent = hanlp.load(hanlp.pretrained.eos.UD_CTB_EOS_MUL)
    texts = examples[field]
    sentences_list = split_sent(texts)
    
    merged_list = []
    for group in sentences_list:
        group = merge_short_sentences(group)
        # Filter out extremely long sentences (more than 100 characters)
        group = [sentence for sentence in group if len(sentence) <= 100]
        merged_list.append(group)

    result = {
        'text': [],
        'source': []
    }
    
    for sentences in merged_list:
        result['text'].extend(sentences)
        result['source'].extend([source] * len(sentences))
    
    return result

# Load and process datasets
datasets = []

def process_dataset(dataset_info):
    name, field = dataset_info['name'], dataset_info['field']
    print(f"Processing {name}...")
    ds = load_dataset(name, split='train' if 'split' not in dataset_info else dataset_info['split'])
    ds = ds.map(
        lambda x: split_into_sentences(x, field, name),
        batched=True,
        remove_columns=ds.column_names,
        batch_size=100000,
        num_proc=32
    )
    ds = ds.flatten()
    return ds

dataset_metadata = [
    {'name': "R5dwMg/zh-wiki-yue-long", 'field': 'text'},
    {'name': "R5dwMg/foodiereview_yue", 'field': 'text'},
    {'name': "raptorkwok/cantonese_sentences", 'field': 'content'},
    {'name': "indiejoseph/cc100-yue", 'field': 'text'},
    {'name': "beyond/chinese_clean_passages_80m", 'field': 'passage', 'split': 'train[:10000000]'},
    {'name': "liswei/Taiwan-Text-Excellence-2B", 'field': 'text'},
    {'name': "jed351/rthk_news", 'field': 'content'},
    {'name': "jed351/shikoto_zh_hk", 'field': 'text'},
]

datasets = [process_dataset(info) for info in dataset_metadata]

# Concatenate all datasets
print("Concatenating datasets...")
combined_dataset = concatenate_datasets(datasets)

# Shuffle the dataset
print("Shuffling the combined dataset...")
combined_dataset = combined_dataset.shuffle(seed=42)

# Split the dataset into train, validation, and test sets
print("Splitting the dataset into train, validation, and test sets...")
dataset_dict = combined_dataset.train_test_split(test_size=20000, seed=42)

# Further split the test set into validation and test
test_valid = dataset_dict['test'].train_test_split(test_size=10000, seed=42)

# Create the final dataset dictionary
final_dataset = DatasetDict({
    'train': dataset_dict['train'],
    'validation': test_valid['train'],
    'test': test_valid['test']
})

print(f"Train set size: {len(final_dataset['train'])}")
print(f"Validation set size: {len(final_dataset['validation'])}")
print(f"Test set size: {len(final_dataset['test'])}")

# Replace the combined_dataset with the split dataset
combined_dataset = final_dataset

# Push to hub
print("Uploading to Hugging Face Hub...")
combined_dataset.push_to_hub("yue_and_zh_sentences", private=True)

print("Process completed successfully!")
