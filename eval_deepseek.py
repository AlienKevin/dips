from spacy.training import Example
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.vocab import Vocab
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


def load_ud(lang='yue_hk'):
    from datasets import load_dataset

    # Load the universal_dependencies dataset from Hugging Face
    dataset = load_dataset('universal-dependencies/universal_dependencies', lang, trust_remote_code=True)

    # Gather all word segmented utterances
    utterances = [[(token, upos_id_to_str(pos)) for token, pos in zip(sentence['tokens'], sentence['upos'])] for sentence in dataset['test']]

    return utterances


if __name__ == "__main__":
    for lang in ['yue', 'zh']:
        print(f'Testing {lang}')
        testing_samples = load_ud(lang + '_hk')

        V = Vocab()
        examples = []
        errors = []

        with open(f"ud_{lang}_outputs_v2/pos_results.jsonl", "r", encoding="utf-8") as f:
            pos_results = [json.loads(line) for line in f]

        for pos_result in tqdm(pos_results):
            input_text = pos_result["input"]
            reference = next((sample for sample in testing_samples if ''.join(token for token, _ in sample) == input_text), None)
            
            if reference is None:
                print(f"Warning: No matching reference found for input: {input_text}")
                continue

            hypothesis = pos_result["result"]
            predicted = Doc(V, words=[x[0] for x in hypothesis], spaces=[False for _ in hypothesis], pos=[x[1] for x in hypothesis])
            target = Doc(V, words=[x[0] for x in reference], spaces=[False for _ in reference], pos=[x[1] for x in reference])
            example = Example(predicted, target)
            examples.append(example)

            # Write erroneous hypothesis along with reference to file
            if reference != hypothesis:
                errors.append({'reference': reference, 'hypothesis': hypothesis})

        with open("errors.jsonl", "w", encoding="utf-8") as f:
            for error in errors:
                f.write(json.dumps(error, ensure_ascii=False) + "\n")

        # Calculate the F1 score using Scorer
        scorer = Scorer()
        results = scorer.score(examples)

        print(f"POS Tagging Accuracy: {results['pos_acc']}")
        print(f"Token Accuracy: {results['token_acc']}")
        print(f"Token F1 Score: {results['token_f']}")
        print(f"Token Precision: {results['token_p']}")
        print(f"Token Recall: {results['token_r']}")
