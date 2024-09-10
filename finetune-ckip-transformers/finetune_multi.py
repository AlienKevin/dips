import json
import random
import datasets

if __name__ == '__main__':
    random.seed(42)

    # read corpuses
    datasets_to_load = ["AlienKevin/hkcancor-multi", "AlienKevin/wiki-yue-long-multi"]
    examples = []

    for dataset_name in datasets_to_load:
        dataset = datasets.load_dataset(dataset_name)['train']
        label_id_to_name = dataset.features["labels"].feature.names

        for entry in dataset:
            words = entry["chars"][:510]
            ner = [label_id_to_name[label_id] for label_id in entry["labels"][:510]]
            examples.append({"words": words, "ner": ner})
    
    random.shuffle(examples)
    
    with open('data/finetune_multi.json', 'w') as train_outfile:
        for entry in examples:
            json.dump(entry, train_outfile, ensure_ascii=False)
            train_outfile.write('\n')
