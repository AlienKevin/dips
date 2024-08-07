from transformers import pipeline

# Initialize the named entity recognition pipeline
ner = pipeline("ner", model='ckiplab/albert-tiny-chinese-ws', tokenizer='bert-base-chinese')

text = "你喺度做緊乜嘢"

# Get the named entities
entities = ner(text)

# Merge entities with field entity == 'B' with entity == 'I' following it into tokens, and show the token list
merged_tokens = []
i = 0
while i < len(entities):
    if entities[i]['entity'] == 'B':
        token = entities[i]['word']
        i += 1
        while i < len(entities) and entities[i]['entity'] == 'I':
            token += entities[i]['word']
            i += 1
        merged_tokens.append(token)
    else:
        merged_tokens.append(entities[i]['word'])
        i += 1

print("Merged Tokens:", merged_tokens)
