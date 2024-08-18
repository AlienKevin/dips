from datasets import Dataset, DatasetDict
import datasets
from typing import List, Tuple
from huggingface_hub import HfApi


all_pos_tags_ud = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

all_pos_tags_ctb = ['AD', 'AS', 'BA', 'CC', 'CD', 'CS', 'DEC', 'DEG', 'DER', 'DEV', 'DT', 'ETC', 'FW', 'IJ', 'JJ', 'LB', 'LC', 'M', 'MSP', 'NN', 'NN-SHORT', 'NOI', 'NR', 'NR-SHORT', 'NT', 'NT-SHORT', 'OD', 'ON', 'P', 'PN', 'PU', 'SB', 'SP', 'URL', 'VA', 'VC', 'VE', 'VV']

def read_tsv(file_path: str) -> List[List[Tuple[str, str]]]:
    sentences = []
    current_sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                if len(line.split('\t')) != 2:
                    print('Broken line:', line)
                    continue
                token, tag = line.split('\t')
                if tag == 'VV-2':
                    tag = 'VV'
                elif tag == 'AS-1':
                    tag = 'AS'
                elif tag == 'MSP-2':
                    tag = 'MSP'
                elif tag == 'X':
                    tag = 'FW'
                current_sentence.append((token, tag))
            elif current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
    
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences

def ctb_to_ud_tags(ctb_tag: str) -> str:
    ctb_to_ud = {
        'AD': 'ADV',
        
        'AS': 'AUX',
        'BA': 'ADP',
        'CC': 'CCONJ',
        'CD': 'NUM',
        'CS': 'SCONJ',

        'DEC': 'PART',
        'DEG': 'PART',
        'DER': 'PART',
        'DEV': 'PART',

        'DT': 'DET',
        'ETC': 'PART',
        'FW': 'X',
        'IJ': 'INTJ',
        'JJ': 'ADJ',
        'LB': 'ADP',
        'LC': 'ADP',
        'M': 'NOUN',
        'MSP': 'PART',

        'NN': 'NOUN',
        'NN-SHORT': 'NOUN',
        'NOI': 'X',
        'NR': 'PROPN',
        'NR-SHORT': 'PROPN',
        'NT': 'NOUN',
        'NT-SHORT': 'NOUN',

        'OD': 'ADJ',
        # Onomatopoeia's tag depends on their usage in context in UD.
        # For now, we just tag them as X.
        # Not a big issue because ON does not occur in CTB 8.0
        'ON': 'X',
        'P': 'ADP',
        'PN': 'PRON',
        'PU': 'PUNCT',
        'SB': 'ADP',
        'SP': 'PART',

        'URL': 'SYM',

        'VA': 'ADJ',
        'VC': 'VERB',
        'VE': 'AUX',
        'VV': 'VERB'
    }
    return ctb_to_ud.get(ctb_tag, 'X')

def create_dataset(file_path: str) -> Dataset:
    pos_tags_ud = datasets.Sequence(
        datasets.features.ClassLabel(names=all_pos_tags_ud)
    )
    pos_tags_ctb = datasets.Sequence(
        datasets.features.ClassLabel(names=all_pos_tags_ctb)
    )

    # Create a Hugging Face dataset
    features = datasets.Features(
        {
            "tokens": datasets.Sequence(datasets.Value("string")),
            "pos_tags_ud": pos_tags_ud,
            "pos_tags_ctb": pos_tags_ctb,
        }
    )

    sentences = read_tsv(file_path)
    
    tokens = []
    pos_tags_ctb = []
    pos_tags_ud = []
    
    for sentence in sentences:
        tokens.append([token for token, _ in sentence])
        pos_tags_ctb.append([tag for _, tag in sentence])
        pos_tags_ud.append([ctb_to_ud_tags(tag) for _, tag in sentence])
    
    return Dataset.from_dict({
        'tokens': tokens,
        'pos_tags_ctb': pos_tags_ctb,
        'pos_tags_ud': pos_tags_ud
    }, features=features)

# Create datasets for train, dev, and test
train_dataset = create_dataset('ctb_outputs/train.tsv')
dev_dataset = create_dataset('ctb_outputs/dev.tsv')
test_dataset = create_dataset('ctb_outputs/test.tsv')

# Combine into a DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'validation': dev_dataset,
    'test': test_dataset
})

api = HfApi()
api.create_repo(repo_id="AlienKevin/ctb8", exist_ok=True)
dataset.push_to_hub("AlienKevin/ctb8")
print("Dataset published successfully to AlienKevin/ctb8.")
