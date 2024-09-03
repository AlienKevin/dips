from datasets import load_dataset
import datasets

all_pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']


def upos_id_to_str(upos_id):
    names=[
        "NOUN",
        "PUNCT",
        "ADP",
        "NUM",
        "SYM",
        "SCONJ",
        "ADJ",
        "PART",
        "DET",
        "CCONJ",
        "PROPN",
        "PRON",
        "X",
        "X",
        "ADV",
        "INTJ",
        "VERB",
        "AUX",
    ]
    return names[upos_id]


def load_and_upload_ud(lang='yue_hk'):
    # Load the universal_dependencies dataset from Hugging Face
    dataset = load_dataset('universal-dependencies/universal_dependencies', lang, trust_remote_code=True)

    def process_ud(example):
        return {
            'sentence': example['text'],
            'sentence_preserved': True,
            'tokens': example['tokens'],
            'pos_tags_ud': [upos_id_to_str(pos) for pos in example['upos']]
        }

    processed_dataset = dataset.map(process_ud, remove_columns=dataset['test'].column_names)

    # Define features
    features = datasets.Features({
        "sentence": datasets.Value("string"),
        "sentence_preserved": datasets.Value("bool"),
        "tokens": datasets.Sequence(datasets.Value("string")),
        "pos_tags_ud": datasets.Sequence(datasets.features.ClassLabel(names=all_pos_tags)),
    })

    # Cast the dataset to the new features
    processed_dataset = processed_dataset.cast(features)

    # Upload to Hugging Face Hub
    processed_dataset.push_to_hub(f"AlienKevin/ud_{lang}")

    return processed_dataset

if __name__ == '__main__':
    load_and_upload_ud('yue_hk')
    load_and_upload_ud('zh_hk')
