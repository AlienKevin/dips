from datasets import load_dataset
from tqdm import tqdm
from spacy.vocab import Vocab
from spacy.training import Example
from spacy.scorer import Scorer
from spacy.tokens import Doc
import hanlp
import torch
import json

def load_and_preprocess_dataset(dataset_name):
    dataset = load_dataset(dataset_name, split='test')
    preprocessed_data = []
    for item in dataset:
        tokens = item['tokens']
        text = ''.join(tokens)
        preprocessed_data.append((text, tokens))
    return preprocessed_data

def evaluate_segmentation(model, dataset_names):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # Load the model
    tok = hanlp.load(model)

    for dataset_name in dataset_names:
        print(f"Evaluating on {dataset_name}")
        
        # Load and preprocess the dataset
        test_data = load_and_preprocess_dataset(dataset_name)

        V = Vocab()
        examples = []
        errors = []

        # Get model predictions
        results = tok([text for text, _ in test_data], devices=[device])

        for i, (text, reference) in enumerate(test_data):
            hypothesis = results[i]

            target = Doc(V, words=reference, spaces=[False] * len(reference))
            predicted = Doc(V, words=hypothesis, spaces=[False] * len(hypothesis))
            example = Example(predicted, target)
            examples.append(example)
            
            if reference != hypothesis:
                errors.append({'reference': ' '.join(reference), 'hypothesis': ' '.join(hypothesis)})

        with open(f'{dataset_name.split("/")[-1]}_seg_errors.jsonl', 'w') as f:
            for error in errors:
                f.write(json.dumps(error, ensure_ascii=False) + '\n')

        scorer = Scorer()
        results = scorer.score(examples)

        print(f"Token F1 Score: {results['token_f']:.4f}")
        print(f"Token Precision: {results['token_p']:.4f}")
        print(f"Token Recall: {results['token_r']:.4f}")
        print()

if __name__ == "__main__":
    model = hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH
    dataset_names = ["AlienKevin/pku-seg", "AlienKevin/msr-seg", "AlienKevin/cityu-seg", "AlienKevin/as-seg", "AlienKevin/ctb8"]
    evaluate_segmentation(model, dataset_names)
