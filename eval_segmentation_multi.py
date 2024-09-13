from datasets import load_dataset
from tqdm import tqdm
from spacy.vocab import Vocab
from spacy.training import Example
from spacy.scorer import Scorer
from spacy.tokens import Doc
import json
import time

def load_and_preprocess_dataset(dataset_name):
    dataset = load_dataset(dataset_name, split='test')
    preprocessed_data = []
    if dataset_name == "AlienKevin/hkcancor-multi":
        for item in dataset:
            chars = item['chars']
            labels = item['labels']
            preprocessed_data.append((''.join(chars), labels))
    else:
        for item in dataset:
            tokens = item['tokens']
            text = ''.join(tokens)
            preprocessed_data.append((text, tokens))
    return preprocessed_data

def evaluate_segmentation(cut, dataset_names):
    results = {}

    total_tokens = 0
    total_time = 0

    for dataset_name in dataset_names:
        print(f"Evaluating on {dataset_name}")
        
        # Load and preprocess the dataset
        test_data = load_and_preprocess_dataset(dataset_name)

        if dataset_name == "AlienKevin/hkcancor-multi":
            # Initialize counters for correct predictions and total predictions
            correct_predictions = 0
            total_predictions = 0

            for text, labels in tqdm(test_data):
                total_tokens += len(text)
                start = time.time()
                predictions = cut(text)
                total_time += time.time() - start

                # Iterate through the characters and labels
                assert len(labels) == len(predictions)

                for label, pred in zip(labels, predictions):
                    pred_label = pred['entity']
                    
                    # Compare prediction with the true label
                    if pred_label == 'DIPS'[label]:
                        correct_predictions += 1
                    total_predictions += 1

            # Calculate accuracy
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")

            results[dataset_name] = {'accuracy': accuracy, 'correct_predictions': correct_predictions, 'total_predictions': total_predictions}

            continue

        V = Vocab()
        examples = []
        errors = []

        for text, reference in tqdm(test_data):
            total_tokens += len(text)
            start = time.time()
            predictions = cut(text)
            total_time += time.time() - start
            
            # Fine segmentation
            hypothesis_fine = []
            current_token = ""

            for pred in predictions:
                if not pred['entity'].startswith('I'):
                    if current_token:
                        hypothesis_fine.append(current_token)
                    current_token = pred['word'].lstrip('##')
                else:
                    current_token += pred['word'].lstrip('##')
            if current_token:
                hypothesis_fine.append(current_token)
            
            if len(''.join(reference)) != len(''.join(hypothesis_fine)):
                print("Hypothesis does not match reference.")
                print("HYP:" + ''.join(hypothesis_fine))
                print("REF:" + ''.join(reference))
                continue
            
            # Coarse segmentation
            hypothesis_coarse = []
            current_token = ""

            for pred in predictions:
                if pred['entity'].startswith('S'):
                    if current_token:
                        hypothesis_coarse.append(current_token)
                    current_token = pred['word'].lstrip('##')
                else:
                    current_token += pred['word'].lstrip('##')
            if current_token:
                hypothesis_coarse.append(current_token)

            # Map coarse tokens to fine tokens
            coarse_to_fine = {}
            char_index = 0
            for coarse_token in hypothesis_coarse:
                start = char_index
                end = start + len(coarse_token)
                fine_tokens = []
                current_length = 0
                fine_char_index = 0
                for fine_token in hypothesis_fine:
                    if fine_char_index < start:
                        fine_char_index += len(fine_token)
                        continue
                    elif fine_char_index > start:
                        break
                    if current_length + len(fine_token) <= len(coarse_token):
                        fine_tokens.append(fine_token)
                        current_length += len(fine_token)
                    else:
                        break
                coarse_to_fine[(start, end)] = fine_tokens
                char_index = end

            # Split/merge matching words in reference according to coarse_to_fine mapping
            new_reference = []
            char_index = 0
            for ref_token in reference:
                start = char_index
                end = start + len(ref_token)
                if (start, end) in coarse_to_fine:
                    new_reference.extend(coarse_to_fine[(start, end)])
                else:
                    new_reference.append(ref_token)
                char_index = end

            # Update reference with the split version
            reference = new_reference
            hypothesis = hypothesis_fine

            target = Doc(V, words=reference, spaces=[False] * len(reference))
            predicted = Doc(V, words=hypothesis, spaces=[False] * len(hypothesis))
            example = Example(predicted, target)
            examples.append(example)
            
            if reference != hypothesis:
                # Fine segmentation
                hypothesis_str = ""

                for pred in predictions:
                    if pred['entity'].startswith('S'):
                        hypothesis_str += " "
                    elif pred['entity'].startswith('D'):
                        hypothesis_str += "-"
                    elif pred['entity'].startswith('P'):
                        hypothesis_str += "|"
                    hypothesis_str += pred['word'].lstrip('##')
                
                hypothesis_str = hypothesis_str.lstrip(' ')
                
                errors.append({'reference': ' '.join(reference), 'hypothesis': hypothesis_str})

        # with open(f'{dataset_name.split("/")[-1]}_seg_errors.jsonl', 'w') as f:
        #     for error in errors:
        #         f.write(json.dumps(error, ensure_ascii=False) + '\n')

        scorer = Scorer()
        scorer_results = scorer.score(examples)

        print(f"Token F1 Score: {scorer_results['token_f']:.4f}")
        print(f"Token Precision: {scorer_results['token_p']:.4f}")
        print(f"Token Recall: {scorer_results['token_r']:.4f}")
        print()

        results[dataset_name] = {'token_f': scorer_results['token_f'], 'token_p': scorer_results['token_p'], 'token_r': scorer_results['token_r']}

    results['total_tokens'] = total_tokens
    results['total_time'] = total_time

    return results


if __name__ == "__main__":
    model_results = {}
    
    ckip_models = [
        "electra_small_hkcancor_multi",
        "electra_small_layers_6_hkcancor_multi",
        "electra_small_layers_5_hkcancor_multi",
        "electra_small_layers_4_hkcancor_multi",
        "electra_small_layers_3_hkcancor_multi",

        "electra_base_hkcancor_multi",
        "electra_large_hkcancor_multi",
        "albert_tiny_chinese_hkcancor_multi",
        "bert_tiny_chinese_hkcancor_multi",

        "electra_small_layers_6_multi",
        "electra_small_layers_6_multi_compressed",
    ]

    dataset_names = ["AlienKevin/hkcancor-multi", "AlienKevin/ud_yue_hk", "AlienKevin/ud_zh_hk", "AlienKevin/cityu-seg"]

    for model_name in ckip_models:
        print(f'Evaluating {model_name}')

        model_path = f"finetune-ckip-transformers/{model_name}"

        model_results[model_name] = {}

        # from transformers import pipeline
        # cut = pipeline("token-classification", model=model_name, device="cpu")

        start = time.time()
        from transformers import AutoModelForTokenClassification
        model_results[model_name]['import_time'] = time.time() - start

        import torch
        from pathlib import Path

        start = time.time()
        if model_name.endswith("compressed"):
            model = AutoModelForTokenClassification.from_pretrained(model_path, torch_dtype=torch.float16).to('cpu')
        else:
            model = AutoModelForTokenClassification.from_pretrained(model_path).to('cpu')
        vocab_path = Path(model_path) / "vocab.txt"
        vocab = {}
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                vocab[line.strip()] = i
        model_results[model_name]['load_time'] = time.time() - start


        def cut(text):
            inputs = torch.tensor([vocab['[CLS]']] + [vocab[char] if char in vocab else vocab['[UNK]'] for char in text.lower()] + [vocab['[SEP]']]).unsqueeze(0)
            with torch.no_grad():
                # squeeze removes the first singleton batch dimension
                # [1, -1] removes the first [CLS] and last [SEP] tokens
                logits = model(input_ids=inputs).logits.squeeze()[1:-1]
                predictions = logits.argmax(dim=-1).tolist()
            return list({"word": token, "entity": "DIPS"[prediction]} for token, prediction in zip(text, predictions))
    
        results = evaluate_segmentation(cut, dataset_names)
        for k, v in results.items():
            model_results[model_name][k] = v


    # from scratch_inference.flax_model import Electra
    # model = Electra()
    # model.load("finetune-ckip-transformers/electra_small_layers_6_multi_compressed")
    # cut = lambda text: model.cut(text)


    gguf_models = [
        "electra.gguf",
        "electra-q8_0.gguf",
        "electra-q4_1.gguf",
        "electra-q4_0.gguf",
    ]

    for model_name in gguf_models:
        model_path = f"bert.cpp/{model_name}"

        print(f'Evaluating {model_name}')

        model_results[model_name] = {}

        import sys
        import os
        module_path = os.path.join(os.path.dirname(__file__), 'bert.cpp')
        sys.path.append(module_path)

        start = time.time()
        from bert_cpp import BertModel
        model_results[model_name]['import_time'] = time.time() - start

        start = time.time()
        model = BertModel(model_path, use_cpu=True)
        model_results[model_name]['load_time'] = time.time() - start

        def cut(text):
            tags = model.cut(text, mode='dips')
            return [{'word': char, 'entity': tag} for char, tag in zip(text, tags)]

        results = evaluate_segmentation(cut, dataset_names)
        for k, v in results.items():
            model_results[model_name][k] = v

    import json
    with open('multi_model_results.json', 'w') as f:
        json.dump(model_results, f, ensure_ascii=False)
