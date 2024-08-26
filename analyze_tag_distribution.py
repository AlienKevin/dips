import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import Counter
import json

# Load the dataset
dataset = load_dataset('AlienKevin/ctb8', split='train')

# Get the feature names for POS tags
pos_tag_names = dataset.features['pos_tags_ud'].feature.names

# Count the occurrences of each POS tag
pos_tags = [pos_tag_names[tag] for example in dataset for tag in example['pos_tags_ud']]
tag_counts = Counter(pos_tags)

# Analyze errors from errors.jsonl
error_tag_counts = Counter()
total_tag_counts = Counter()

with open('errors.jsonl', 'r', encoding='utf-8') as f:
    for line_index, line in enumerate(f):
        error = json.loads(line)
        reference = error['REF'].split()
        hypothesis = error['HYP'].split()
        
        for i, (ref_word, ref_tag) in enumerate(zip(reference, dataset[line_index]['pos_tags_ud'])):
            if ref_word not in hypothesis:
                # Count the tag of incorrectly segmented token
                error_tag_counts[pos_tag_names[ref_tag]] += 1
            total_tag_counts[pos_tag_names[ref_tag]] += 1

# Combine the data for plotting
all_tags = set(tag_counts.keys()) | set(error_tag_counts.keys())
sorted_tags = sorted(all_tags, key=lambda x: tag_counts.get(x, 0) + error_tag_counts.get(x, 0), reverse=True)

# Prepare data for plotting
overall_counts = [tag_counts.get(tag, 0) for tag in sorted_tags]
error_counts = [error_tag_counts.get(tag, 0) / total_tag_counts.get(tag, 1) * tag_counts.get(tag, 1) for tag in sorted_tags]

# Create a side-by-side bar plot
fig, ax = plt.subplots(figsize=(15, 8))

x = range(len(sorted_tags))
width = 0.35

ax.bar([i - width/2 for i in x], overall_counts, width, label='Overall Distribution', alpha=0.8)
ax.bar([i + width/2 for i in x], error_counts, width, label='Error Distribution', alpha=0.8)

ax.set_ylabel('Frequency')
ax.set_title('Distribution of POS Tags: Overall vs Errors')
ax.set_xticks(x)
ax.set_xticklabels(sorted_tags, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.show()

# Print the distributions
print("Overall POS Tag Distribution:")
for tag, count in tag_counts.most_common():
    print(f"{tag}: {count}")

print("\nDistribution of POS Tags for Incorrectly Segmented Tokens:")
for tag, count in error_tag_counts.most_common():
    print(f"{tag}: {count}")
