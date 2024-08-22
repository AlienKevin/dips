# pip install hanlp -U

import hanlp
from datasets import load_dataset, DatasetDict
import torch
from multiprocess import set_start_method

if __name__ == '__main__':
    set_start_method('spawn')
    torch.multiprocessing.set_start_method('spawn')
    
    dataset = load_dataset("beyond/chinese_clean_passages_80m", split="train")
    
    # Define a function to tokenize sentences
    def tokenize_sentence(example):
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        tok_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH, devices=[device])
        tokens = tok_fine(example["passage"])
        return {"tokens": tokens}
    
    # Apply the tokenization function
    dataset = dataset.select(range(10000000)).map(tokenize_sentence, batched=True, remove_columns=["passage"], batch_size=10000, num_proc=10)
    
    # Split the dataset into train, validation, and test sets
    train_testvalid = dataset.train_test_split(test_size=20000*2, seed=42)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
    
    dataset = DatasetDict({
        'train': train_testvalid['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    })
    
    print(f"Train set: {len(dataset['train'])} samples")
    print(f"Validation set: {len(dataset['validation'])} samples")
    print(f"Test set: {len(dataset['test'])} samples")
    
    # Upload the dataset to the Hugging Face Hub
    dataset.push_to_hub("AlienKevin/genius-seg")
    
    print(f"Uploaded segmented sentences to AlienKevin/genius-seg on the Hugging Face Hub")
