from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

model_id = "../finetune-ckip-transformers/electra_small_layers_6_multi_compressed"

# Must specify torch_dtype=torch.float16 to load the fp16 model as fp16 in memory
model = AutoModelForTokenClassification.from_pretrained(model_id, torch_dtype=torch.float16).to('cpu')
tokenizer = AutoTokenizer.from_pretrained(model_id)

inputs = tokenizer(list("阿李明like佢"), return_tensors="pt", is_split_into_words=True)

with torch.no_grad():
    print(inputs)
    logits = model(**inputs).logits

print(logits)
