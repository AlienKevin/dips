import random
from spacy.training import Example
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.vocab import Vocab
from transformers import pipeline
from tqdm import tqdm
import json

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


def load_ud_yue():
    from datasets import load_dataset

    # Load the universal_dependencies dataset from Hugging Face
    dataset = load_dataset('universal-dependencies/universal_dependencies', 'yue_hk', trust_remote_code=True)

    # Gather all word segmented utterances
    utterances = [[(token, upos_id_to_str(pos)) for token, pos in zip(sentence['tokens'], sentence['upos'])] for sentence in dataset['test']]

    return utterances

def load_cc100_yue_test():
    from datasets import load_dataset

    # Load the cc100 dataset from Hugging Face
    dataset = load_dataset('AlienKevin/cc100-yue-tagged')

    label_names = dataset["test"].features["pos_tags_ud"].feature.names
    id2label = { i:k for i, k in enumerate(label_names) }

    utterances = []
    for item in dataset['test']:
        utterances.append([(token, id2label[pos]) for token, pos in zip(item['tokens'], item['pos_tags_ud'])])

    return utterances

# Patches https://github.com/jacksonllee/pycantonese/issues/48
def patch_pycantonese_tag_bug(tag):
    if tag == "V":
        return "VERB"
    else:
        return tag


if __name__ == "__main__":
    # model_name = "AlienKevin/bert_base_cantonese_pos_hkcancor"
    model_name = "AlienKevin/electra_hongkongese_small_pos_hkcancor"

    sample_size = 100

    testing_samples = load_ud_yue()
    # testing_samples = load_cc100_yue_test()

    random.seed(42)
    random.shuffle(testing_samples)
    testing_samples = testing_samples[:sample_size]

    tagger = pipeline(
        "token-classification",
        model_name,
        grouped_entities=True,
    )

    V = Vocab()
    examples = []
    errors = []

    for sample in tqdm(testing_samples):
        reference = sample
        hypothesis = [(item['word'].removeprefix('##').replace(' ', ''), item['entity_group']) for item in tagger(''.join(x[0] for x in sample))]
        predicted = Doc(V, words=[x[0] for x in hypothesis], spaces=[False for _ in hypothesis], pos=[x[1] for x in hypothesis])
        target = Doc(V, words=[x[0] for x in reference], spaces=[False for _ in reference], pos=[patch_pycantonese_tag_bug(x[1]) for x in reference])
        example = Example(predicted, target)
        examples.append(example)

        # Write erroneous hypothesis along with reference to file
        if reference != hypothesis:
            errors.append({'reference': reference, 'hypothesis': hypothesis})

    with open("errors.jsonl", "w") as f:
        for error in errors:
            f.write(json.dumps(error, ensure_ascii=False) + "\n")

    # Calculate the F1 score using Scorer
    scorer = Scorer()
    results = scorer.score(examples)

    print(f"POS Tagging Accuracy: {results['pos_acc']}")
    print(f"Token F1 Score: {results['token_f']}")
    print(f"Token Precision: {results['token_p']}")
    print(f"Token Recall: {results['token_r']}")
