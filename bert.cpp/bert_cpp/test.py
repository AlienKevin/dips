from model import BertModel

model = BertModel(f'../electra.gguf', use_cpu=True)
print(model.cut("阿張先生", mode='fine'))
print(model.cut("阿張先生", mode='coarse'))
print(model.cut("阿張先生", mode='dips'))
