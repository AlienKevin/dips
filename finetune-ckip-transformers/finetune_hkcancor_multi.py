import json
import random
import re
import datasets

if __name__ == '__main__':
    random.seed(42)

    # read corpus
    hkcancor = datasets.load_dataset("AlienKevin/hkcancor-multi")['train']

    label_id_to_name = hkcancor.features["labels"].feature.names

    examples = []
    for entry in hkcancor:
        examples.append({"words": entry["chars"], "ner": [label_id_to_name[label_id] for label_id in entry["labels"]]})
    random.shuffle(examples)
    with open('data/finetune_hkcancor_multi.json', 'w') as train_outfile:
        for entry in examples:
            json.dump(entry, train_outfile, ensure_ascii=False)
            train_outfile.write('\n')
