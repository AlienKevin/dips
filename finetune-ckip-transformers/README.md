# finetune-ckip-transformers
Create training files to fine-tune Hugging Face models to use with [CKIP Transformers](https://github.com/ckiplab/ckip-transformers). This is part of creating Hongkongese models using the same method.

## Overview

Hugging Face provides examples for [token classification](https://huggingface.co/docs/transformers/main/en/task_summary#token-classification). CKIP Transformers uses BI encoding to indicate word segmentation. For a sentence 點解 啊 ?, the line in the file looks like `{"words": ["點", "解", "啊", "?"], "ner": ["B", "I", "B", "B"]}`

Fine-tuning with this training data creates a model that can be loaded and used by CKIP Transformers (for non-bert models, some code changes to use different tokenizers will be needed).

## Instructions
1. Install [PyCantonese](https://pycantonese.org/) to use the HKCanCor dataset
2. Download training data from [The Second International Chinese Word Segmentation Bakeoff](http://sighan.cs.uchicago.edu/bakeoff2005/) and place cityu_training.utf8 and/or as_training.utf8 in /data
3. Run finetune_hkcancor.py and finetune_cityu.py (uncomment some lines if as_training.utf8 is used). finetune_hkcancor.json and finetune_cityu.json will be created
4. Merge/shuffle the created files if needed
5. Install Hugging Face Transformers from [source](https://github.com/huggingface/transformers/)
6. Go to transformers/examples/pytorch/token-classification/
7. Run fine-tuning with the training file `python run_ner.py --model_name_or_path toastynews/electra-hongkongese-base-discriminator --train_file finetune_hkcancor.json --output_dir tn_electra_base_hkcancor --do_train`

## Versions
The following software versions were used.
* pycantonese 3.4.0