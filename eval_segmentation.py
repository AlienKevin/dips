import torch
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm
from spacy.vocab import Vocab
from spacy.training import Example
from spacy.scorer import Scorer
from spacy.tokens import Doc
import json

def load_and_preprocess_dataset(dataset_name):
    dataset = load_dataset(dataset_name, split='test')
    preprocessed_data = []
    for item in dataset:
        tokens = item['tokens']
        text = ''.join(tokens)
        preprocessed_data.append((text, tokens))
    return preprocessed_data

def evaluate_segmentation(model_name, dataset_names):
    # Load the model
    nlp = pipeline("token-classification", model=model_name, device="cpu")

    for dataset_name in dataset_names:
        print(f"Evaluating on {dataset_name}")
        
        # Load and preprocess the dataset
        test_data = load_and_preprocess_dataset(dataset_name)

        V = Vocab()
        examples = []
        errors = []

        for text, reference in tqdm(test_data):
            # Get model predictions
            predictions = nlp(text)
            
            # Convert predictions to tokens
            hypothesis = []
            current_token = ""

            for pred in predictions:
                if pred['entity'] == 'B':
                    if current_token:
                        hypothesis.append(current_token)
                    current_token = pred['word'].lstrip('##')
                else:
                    current_token += pred['word'].lstrip('##')
            if current_token:
                hypothesis.append(current_token)

            if len(''.join(reference)) != len(''.join(hypothesis)):
                print("Hypothesis does not match reference.")
                print("HYP:" + ''.join(hypothesis))
                print("REF:" + ''.join(reference))
                continue

            target = Doc(V, words=reference, spaces=[False] * len(reference))
            predicted = Doc(V, words=hypothesis, spaces=[False] * len(hypothesis))
            example = Example(predicted, target)
            examples.append(example)
            
            if reference != hypothesis:
                errors.append({'reference': reference, 'hypothesis': hypothesis})

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
    # model_name = "toastynews/electra-hongkongese-small-hkt-ws"
    model_name = "finetune-ckip-transformers/electra_small_hkcancor"
    dataset_names = ["AlienKevin/ud_yue_hk", "AlienKevin/ud_zh_hk", "AlienKevin/cityu-seg", "AlienKevin/as-seg"]
    evaluate_segmentation(model_name, dataset_names)
