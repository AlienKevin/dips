import json
import random
import re

reg = re.compile(r'[a-zA-Z\'-]+')

# read corpus
examples = []
# with open("data/as_training.utf8") as fp:
with open("data/cityu_training.utf8") as fp:    
    lines = fp.readlines()
    for line in lines:
        # create lines
        sent = []
        words = []
        ners = []    
        was_eng = False
        # tokens = line.strip().split('　') # as
        tokens = line.strip().split(' ') # cityu
        for token in tokens:
            this_word = token
            if len(this_word) > 1 and reg.match(this_word):
                if was_eng:
                    sent += [' ']
                    word = [' ']
                    ner = ["B"]
                    words.extend(word)
                    ners.extend(ner)    
                
            if len(this_word) > 1 and (reg.match(this_word[-1:]) or reg.match(this_word[-2:])):
                was_eng = True                       
            else:                        
                was_eng = False           
            sent += [this_word]        
            word = [*this_word]
            ner = ["B"]
            if len(word) > 1:
                ner2 = ["I"] * (len(word)-1)
                ner.extend(ner2)
            words.extend(word)
            ners.extend(ner)      
            
        examples.append({"words":words, "ner": ners})

random.shuffle(examples)
# with open('data/finetune_as.json', 'w') as train_outfile:
with open('data/finetune_cityu.json', 'w') as train_outfile:    
    for entry in examples:
        json.dump(entry, train_outfile, ensure_ascii=False)
        train_outfile.write('\n')
