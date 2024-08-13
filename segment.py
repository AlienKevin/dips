from datasets import load_dataset
from tqdm import tqdm
from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy.training import Example
from spacy.scorer import Scorer
import json

def load_cc100():
    dataset = load_dataset('AlienKevin/cc100-yue-tagged', split='train')
    words = set()
    for item in dataset:
        words.update(item['tokens'])
    return words

def longest_match_segmentation(text, words):
    segments = []
    i = 0
    while i < len(text):
        longest_match = text[i]
        for j in range(i + 1, len(text) + 1):
            if text[i:j] in words:
                longest_match = text[i:j]
        segments.append(longest_match)
        i += len(longest_match)
    return segments

def load_ud_yue(split='test'):
    dataset = load_dataset('universal-dependencies/universal_dependencies', 'yue_hk', split=split)
    return [sentence['tokens'] for sentence in dataset]

def evaluate_segmentation(reference_dataset, segmentation_func, words, error_file):
    V = Vocab()
    examples = []
    errors = []
    for reference in tqdm(reference_dataset):
        text = ''.join(reference)
        hypothesis = segmentation_func(text, words)
        target = Doc(V, words=reference, spaces=[False] * len(reference))
        predicted = Doc(V, words=hypothesis, spaces=[False] * len(hypothesis))
        example = Example(predicted, target)
        examples.append(example)
        
        if reference != hypothesis:
            errors.append({'reference': reference, 'hypothesis': hypothesis})

    with open(error_file, 'w') as f:
        for error in errors:
            f.write(json.dumps(error, ensure_ascii=False) + '\n')

    scorer = Scorer()
    results = scorer.score(examples)
    return results

if __name__ == "__main__":
    print("Loading CC100 dataset...")
    cc100_words = load_cc100()

    print("Loading UD Yue dataset...")
    ud_yue_dataset = load_ud_yue()

    print("Loading CC100 test dataset...")
    cc100_test_dataset = load_dataset('AlienKevin/cc100-yue-tagged', split='test')
    cc100_test = [item['tokens'] for item in cc100_test_dataset]

    print("Evaluating on UD Yue...")
    ud_yue_results = evaluate_segmentation(ud_yue_dataset, longest_match_segmentation, cc100_words, 'ud_yue_seg_errors.jsonl')

    print("Evaluating on CC100 test set...")
    cc100_results = evaluate_segmentation(cc100_test, longest_match_segmentation, cc100_words, 'cc100_seg_errors.jsonl')

    print("UD Yue Results:")
    print(f"Token F1 Score: {ud_yue_results['token_f']}")
    print(f"Token Precision: {ud_yue_results['token_p']}")
    print(f"Token Recall: {ud_yue_results['token_r']}")

    print("\nCC100 Test Results:")
    print(f"Token F1 Score: {cc100_results['token_f']}")
    print(f"Token Precision: {cc100_results['token_p']}")
    print(f"Token Recall: {cc100_results['token_r']}")
