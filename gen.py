import json
import random
import time
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI
import os
from datasets import load_dataset

base_url = "https://api.deepseek.com"
with open('deepseek_api_key.txt', 'r') as file:
    api_key = file.read().strip()
model_id = 'deepseek-chat'
max_workers = 50

client = OpenAI(api_key=api_key, base_url=base_url)

valid_pos_tags = {"ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"}

def segment_words(pos_prompt, input_sentence):
    attempts = 0
    while True:
        result = None
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{
                            "role": "system",
                            "content": pos_prompt
                          },
                          {
                            "role": "user",
                            "content": input_sentence
                          }],
                response_format={
                    'type': 'json_object'
                },
                max_tokens=2000,
                temperature=0,
                stream=False
            )
            result = response.choices[0].message.content
            try:
                result_json = json.loads(result)
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse JSON response: {str(e)}")
            if "pos_tagged_words" in result_json and isinstance(result_json["pos_tagged_words"], list) and \
                all(isinstance(item, list) and len(item) == 2 and (all(isinstance(sub_item, str) for sub_item in item)) for item in result_json["pos_tagged_words"]):
                concatenated_words = "".join([word for word, pos in result_json["pos_tagged_words"]])
                if concatenated_words == input_sentence:
                    for word, pos in result_json["pos_tagged_words"]:
                        if pos not in valid_pos_tags:
                            raise Exception(f"Invalid POS tag '{pos}' in the result")
                    return result_json["pos_tagged_words"]
                else:
                    raise Exception(f"Segmentation result does not match the input sentence")
            else:
                raise Exception(f"Invalid segmentation result format")
        except Exception as e:
            time.sleep(1)
            attempts += 1
            if attempts >= 3:
                return {'error': str(e), 'result': result}


def load_hkcancor(min_tokens, max_tokens):
    dataset = load_dataset("nanyang-technological-university-singapore/hkcancor", trust_remote_code=True)

    label_names = dataset["train"].features["pos_tags_ud"].feature.names
    id2name = {id: name for id, name in enumerate(label_names)}

    utterances = [[[token, id2name[pos]] for token, pos in zip(utterance['tokens'], utterance['pos_tags_ud'])] for utterance in dataset['train'] if len(utterance['tokens']) <= max_tokens and len(utterance['tokens']) >= min_tokens]

    return utterances


def generate_prompt(prompt_version, segmentation_given):
    with open(f'prompt_v{prompt_version}.txt', 'r') as file:
        prompt_prefix = file.read()

    utterances = load_hkcancor(min_tokens=5, max_tokens=20)

    random.seed(42)
    utterances = random.sample(utterances, 10)

    in_context_samples = utterances
    
    # Format in-context samples for the prompt
    in_context_prompt = "\n\n".join([
        f'EXAMPLE INPUT SENTENCE:\n{(" " if segmentation_given else "").join([word for word, pos in sample])}\n\nEXAMPLE JSON OUTPUT:\n{json.dumps({"pos_tagged_words": [[word, pos] for word, pos in sample]}, ensure_ascii=False)}'
        for sample in in_context_samples
    ])

    # Update the word segmentation prompt with in-context samples
    pos_prompt = f"{prompt_prefix}\n\n{in_context_prompt}"
    
    return pos_prompt


if __name__ == "__main__":
    prompt_version = 2

    pos_prompt = generate_prompt(prompt_version=prompt_version, segmentation_given=False)

    print(pos_prompt)

    test_samples = load_dataset("indiejoseph/cc100-yue")['train']
    test_samples = [sample['text'] for sample in test_samples if len(sample['text']) <= 100]

    # Create the output directory if it doesn't exist
    output_dir = f'outputs_v{prompt_version}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pos_results_path = f'{output_dir}/pos_results.jsonl'
    pos_errors_path = f'{output_dir}/pos_errors.jsonl'

    existing_samples = set()

    if os.path.exists(pos_results_path):
        with open(pos_results_path, 'r', encoding='utf-8') as file:
            for line in file:
                result = json.loads(line)
                existing_samples.add(result['input'])

    if os.path.exists(pos_errors_path):
        with open(pos_errors_path, 'r', encoding='utf-8') as file:
            for line in file:
                error_result = json.loads(line)
                existing_samples.add(error_result['input'])

    initial_sample_count = len(test_samples)
    test_samples = [sample for sample in test_samples if sample not in existing_samples]
    removed_sample_count = initial_sample_count - len(test_samples)

    print(f"Number of samples already generated: {removed_sample_count}")

    with open(pos_results_path, 'a+', encoding='utf-8') as file, open(pos_errors_path, 'a+', encoding='utf-8') as error_file:
        lock = Lock()
        def process_sample(input_sentence):
            pos_result = segment_words(pos_prompt, input_sentence.replace(" ", ""))
            if 'error' in pos_result:
                print(f"POS tagging failed for sentence: {input_sentence}")
                print(f"Error: {pos_result['error']}")
                error_result = {
                    "input": input_sentence,
                    "error": pos_result['error'],
                    "result": pos_result['result']
                }
                with lock:
                    error_file.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                    error_file.flush()
            else:
                result = {
                    "input": input_sentence,
                    "result": pos_result
                }
                with lock:
                    file.write(json.dumps(result, ensure_ascii=False) + '\n')
                    file.flush()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(process_sample, test_samples), total=len(test_samples)))
