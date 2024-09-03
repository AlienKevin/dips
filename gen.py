import json
import random
import time
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI
import os
import argparse
import re
from datasets import load_dataset
import opencc

base_url = "https://api.deepseek.com"
with open('deepseek_api_key.txt', 'r') as file:
    api_key = file.read().strip()
model_id = 'deepseek-chat'
max_workers = 1000

client = OpenAI(api_key=api_key, base_url=base_url)

valid_pos_tags = {"ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"}


class Sim:
    def __init__(self, text=''):
        self.unigrams = set(text)
        self.bigrams = set(self.get_bigrams(text))
    
    def get_bigrams(self, text):
        return [text[i:i+2] for i in range(len(text)-1)]
    
    def distance(self, other):
        unigram_distance = self.compare_sets(self.unigrams, other.unigrams)
        bigram_distance = self.compare_sets(self.bigrams, other.bigrams)
        return (unigram_distance + bigram_distance) / 2
    
    def compare_sets(self, set1, set2):
        union = set1 | set2
        intersection = set1 & set2
        return 1 - (len(intersection) / len(union)) if union else 0
    
    def similarity(self, other):
        return 1 - self.distance(other)


def segment_words(prompt_prefix, input_sentence, in_context_samples, presegmented):
    if not args.presegmented:
        input_sentence = input_sentence.replace(" ", "")
    if len(in_context_samples) == 10:
        samples = generate_in_context_prompt(list(in_context_samples.values()), presegmented=presegmented)
    else:
        # Convert input_sentence to Sim
        input_simhash = Sim(input_sentence)
        
        # Calculate Sim distances for all in_context_samples
        distances = [(sample, input_simhash.distance(sample)) for sample in in_context_samples]
        # Sample top 10 with randomness, where the weight is the inverse of the distance
        import math
        weights = [math.exp(-distance) for _, distance in distances]  # Exponential sampling
        random.seed(42)
        top_10_samples = random.choices(distances, weights=weights, k=min(10, len(distances)))

        # Extract just the samples from the (sample, distance) tuples
        closest_samples = [in_context_samples[sample] for sample, _ in top_10_samples]
        
        # Generate in-context prompt using the closest samples
        samples = generate_in_context_prompt(closest_samples, presegmented=presegmented)
    
    # Write prompt and samples to a file
    with open('prompt_sample.txt', 'w', encoding='utf-8') as f:
        f.write(f'{prompt_prefix}\n\n{samples}')

    attempts = 0
    while True:
        result = None
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{
                            "role": "system",
                            "content": f'{prompt_prefix}\n\n{samples}'
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
                if concatenated_words.replace(" ", "") == input_sentence.replace(" ", ""):
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


def load_prompt_dataset(dataset_name, min_tokens, max_tokens):
    if dataset_name == 'hkcancor':
        dataset_name = 'nanyang-technological-university-singapore/hkcancor'
        subset_name=None
        split_name='train'
        pos_tag_name = 'pos_tags_ud'
    elif dataset_name == 'zh_pud':
        dataset_name = 'universal-dependencies/universal_dependencies'
        subset_name = 'zh_pud'
        split_name='test'
        pos_tag_name = 'upos'


    dataset = load_dataset(path=dataset_name, name=subset_name, trust_remote_code=True)

    label_names = dataset[split_name].features[pos_tag_name].feature.names
    id2name = {id: name for id, name in enumerate(label_names)}

    utterances = [[[token, id2name[pos]] for token, pos in zip(utterance['tokens'], utterance[pos_tag_name])] for utterance in dataset[split_name] if len(utterance['tokens']) <= max_tokens and len(utterance['tokens']) >= min_tokens]

    return utterances


def cut_sent(para):
    """
    Cut a paragraph into sentences.

    This function splits a given paragraph into sentences based on various punctuation marks
    and formatting rules specific to Chinese and English text.

    Args:
        para (str): The input paragraph to be segmented.

    Returns:
        list: A list of segmented sentences.

    Examples:
        >>> cut_sent("你好！我是小明。你呢？")
        ['你好！', '我是小明。', '你呢？']

        >>> cut_sent("他說：「今天天氣真好。」我同意。")
        ['他說：「今天天氣真好。」', '我同意。']

        >>> cut_sent("他說：“今天天氣真好。”我同意。")
        ['他說：“今天天氣真好。”', '我同意。']

        >>> cut_sent("這是一個句子...還有一個。")
        ['這是一個句子...', '還有一個。']

        >>> cut_sent("第一句。第二句！第三句？")
        ['第一句。', '第二句！', '第三句？']

        >>> cut_sent("這是…一個...長句子。")
        ['這是…', '一個...', '長句子。']

        >>> cut_sent("你好")
        ['你好']
    """
    para = re.sub(r'([。！!？?])([^”’」』])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub(r'(\.{2,})([^”’」』])', r"\1\n\2", para)  # 英文省略号
    para = re.sub(r'(…{1,})([^”’」』])', r"\1\n\2", para)  # 中文省略号
    para = re.sub(r'([.。！!？?][”’」』])([^，,。.！!？?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


def cut_utterance(utterance):
    """
    Cut an utterance into sentences and align the POS tags.

    Args:
        utterance (list): A list of (token, pos) tuples.

    Returns:
        list: A list of segmented utterances, where each utterance is a list of (token, pos) tuples.

    Example:
        >>> utterance = [('你好', 'INTJ'), ('！', 'PUNCT'), ('我', 'PRON'), ('是', 'AUX'), ('小明', 'PROPN'), ('。', 'PUNCT'), ('你', 'PRON'), ('呢', 'PART'), ('？', 'PUNCT')]
        >>> cut_utterance(utterance)
        [[('你好', 'INTJ'), ('！', 'PUNCT')], [('我', 'PRON'), ('是', 'AUX'), ('小明', 'PROPN'), ('。', 'PUNCT')], [('你', 'PRON'), ('呢', 'PART'), ('？', 'PUNCT')]]

        >>> utterance = [('你好', 'INTJ')]
        >>> cut_utterance(utterance)
        [[('你好', 'INTJ')]]

        >>> utterance = [('...', 'PUNCT')]
        >>> cut_utterance(utterance)
        [[('...', 'PUNCT')]]
    """
    # Get all tokens from the utterance
    tokens = ''.join([token for token, _ in utterance])

    # Cut the tokens into sentences
    sentences = cut_sent(tokens)

    # Align the sentences back to the utterance
    segmented_utterances = []
    start_index = 0
    for sentence in sentences:
        end_index = start_index
        current_sentence = []
        while end_index < len(utterance):
            token, pos = utterance[end_index]
            current_sentence.append((token, pos))
            end_index += 1
            if ''.join(token for token, _ in current_sentence) == sentence:
                break
        
        if end_index == len(utterance) and len(sentences) == 1:
            # If there's only one sentence and we've reached the end, include all tokens
            segmented_utterances.append(utterance[start_index:])
        else:
            # Ensure the last token is a punctuation mark
            if pos != 'PUNCT':
                end_index -= 1
                current_sentence.pop()
            segmented_utterances.append(current_sentence)
        
        start_index = end_index

    segmented_utterances = [utterance for utterance in segmented_utterances if utterance]

    return segmented_utterances


def generate_in_context_prompt(utterances, presegmented=False):
    in_context_samples = utterances
    
    # Format in-context samples for the prompt
    in_context_prompt = "\n\n".join([
        f'EXAMPLE INPUT SENTENCE:\n{(" " if presegmented else "").join([word for word, pos in sample])}\n\nEXAMPLE JSON OUTPUT:\n{json.dumps({"pos_tagged_words": [[word, pos] for word, pos in sample]}, ensure_ascii=False)}'
        for sample in in_context_samples
    ])

    return in_context_prompt


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, choices=['cc100_yue', 'lihkg', 'wiki_yue_long', 'genius', 'ud_yue', 'ud_zh'], required=True)
    args.add_argument('--prompt_dataset', type=str, choices=['hkcancor', 'zh_pud'], required=True)
    args.add_argument('--prompt_language', type=str, choices=['zh', 'yue'], required=True)
    args.add_argument('--prompt_script', type=str, choices=['simplified', 'traditional'], required=True)
    args.add_argument('--prompt_version', type=int, default=2, required=True)
    args.add_argument('--selective_in_context', action='store_true')
    args.add_argument('--presegmented', action='store_true')
    args = args.parse_args()

    with open(f'data/prompt_v{args.prompt_version}_{args.prompt_language}.txt', 'r') as file:
        prompt_prefix = file.read()

    if args.prompt_script == 'simplified':
        prompt_prefix = opencc.OpenCC('t2s').convert(prompt_prefix)

    if args.selective_in_context:
        in_context_utterances = load_prompt_dataset(dataset_name=args.prompt_dataset, min_tokens=0, max_tokens=99999)
        in_context_utterances = [segment for utterance in in_context_utterances for segment in cut_utterance(utterance)]
        in_context_utterances = [utterance for utterance in in_context_utterances if len(utterance) >= 5 and len(utterance) <= 40]
    else:
        in_context_utterances = load_prompt_dataset(dataset_name=args.prompt_dataset, min_tokens=5, max_tokens=20)
        random.seed(42)
        in_context_utterances = random.sample(in_context_utterances, 10)
    
    # Compute Sim for utterances and store as hkcancor_hash_table
    in_context_samples = {}

    if args.prompt_script == 'simplified':
        converter = opencc.OpenCC('t2s')

    for utterance in in_context_utterances:
        if args.prompt_script == 'simplified':
            utterance = [(converter.convert(word), pos) for word, pos in utterance]

        # Convert the utterance to a string representation
        utterance_str = ''.join([word for word, pos in utterance])

        # Compute Sim for the utterance
        hash_value = Sim(utterance_str)
        # Store the hash value with the utterance as the key
        in_context_samples[hash_value] = utterance

    if args.dataset == 'cc100_yue':
        test_samples = load_dataset("indiejoseph/cc100-yue")['train']
        test_samples = [sample['text'] for sample in test_samples if (len(sample['text']) <= 100 and '嘅發音' not in sample['text'] and 'Hotels.com' not in sample['text'])]
    elif args.dataset == 'lihkg':
        test_samples = load_dataset("raptorkwok/cantonese_sentences")['train']
        test_samples.shuffle(seed=42)
        test_samples = [sample['content'] for sample in test_samples if len(sample['content']) <= 100][:200000]
    elif args.dataset == 'wiki_yue_long':
        test_samples = load_dataset("R5dwMg/zh-wiki-yue-long")['train']
        test_samples.shuffle(seed=42)
        test_samples = [sample['text'] for sample in test_samples if len(sample['text']) <= 200]
    elif args.dataset == 'genius':
        test_samples = load_dataset("beyond/chinese_clean_passages_80m")['train']
        test_samples = test_samples.select(range(50000))['passage']
    elif args.dataset == 'ud_yue':
        test_samples = load_dataset('universal-dependencies/universal_dependencies', 'yue_hk', trust_remote_code=True)['test']
        test_samples = [' '.join(tokens) for tokens in test_samples['tokens']]
    elif args.dataset == 'ud_zh':
        test_samples = load_dataset('universal-dependencies/universal_dependencies', 'zh_hk', trust_remote_code=True)['test']
        test_samples = [' '.join(tokens) for tokens in test_samples['tokens']]

    # Create the output directory if it doesn't exist
    output_dir = f'{args.dataset}_outputs_v{args.prompt_version}{"_presegmented" if args.presegmented else ""}'
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
            pos_result = segment_words(prompt_prefix, input_sentence, in_context_samples, presegmented=args.presegmented)
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
