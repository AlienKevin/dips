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

def evaluate_segmentation(cut, dataset_names):
    for dataset_name in dataset_names:
        print(f"Evaluating on {dataset_name}")
        
        # Load and preprocess the dataset
        test_data = load_and_preprocess_dataset(dataset_name)

        V = Vocab()
        examples = []
        errors = []

        for text, reference in tqdm(test_data):
            # Get model predictions
            predictions = cut(text)
            
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
    # model_name = "finetune-ckip-transformers/electra_small_hkcancor_multi"
    # model_name = "finetune-ckip-transformers/electra_small_layers_6_hkcancor_multi"
    # model_name = "finetune-ckip-transformers/electra_small_layers_5_hkcancor_multi"
    # model_name = "finetune-ckip-transformers/electra_small_layers_4_hkcancor_multi"
    # model_name = "finetune-ckip-transformers/electra_small_layers_3_hkcancor_multi"

    # model_name = "finetune-ckip-transformers/electra_small_layers_6_multi"
    model_name = "finetune-ckip-transformers/electra_small_layers_6_multi_compressed"

    # model_name = "finetune-ckip-transformers/electra_base_hkcancor_multi"
    # model_name = "finetune-ckip-transformers/electra_large_hkcancor_multi"
    # model_name = "finetune-ckip-transformers/albert_tiny_chinese_hkcancor_multi"
    # model_name = "finetune-ckip-transformers/bert_tiny_chinese_hkcancor_multi"

    dataset_names = ["AlienKevin/ud_yue_hk", "AlienKevin/ud_zh_hk", "AlienKevin/cityu-seg", "AlienKevin/as-seg"]

    # from transformers import pipeline
    # cut = pipeline("token-classification", model=model_name, device="cpu")

    from transformers import AutoModelForTokenClassification, AutoTokenizer
    import torch
    from pathlib import Path
    model = AutoModelForTokenClassification.from_pretrained(model_name, torch_dtype=torch.float16).to('cpu')
    vocab_path = Path(model_name) / "vocab.txt"
    vocab = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            vocab[line.strip()] = i

    def cut(text):
        inputs = torch.tensor([vocab['[CLS]']] + [vocab[char] if char in vocab else vocab['[UNK]'] for char in text.lower()] + [vocab['[SEP]']]).unsqueeze(0)
        with torch.no_grad():
            # squeeze removes the first singleton batch dimension
            # [1, -1] removes the first [CLS] and last [SEP] tokens
            logits = model(input_ids=inputs).logits.squeeze()[1:-1]
            predictions = logits.argmax(dim=-1).tolist()
        return list({"word": token, "entity": "DIPS"[prediction]} for token, prediction in zip(text, predictions))

    # from scratch_inference.flax_model import Electra
    # model = Electra()
    # model.load("finetune-ckip-transformers/electra_small_layers_6_multi_compressed")
    # cut = lambda text: model.cut(text)

    evaluate_segmentation(cut, dataset_names)
