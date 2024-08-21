class Vocab:
    def __init__(self, token2id_map):
        self.token2id_map = token2id_map
        self.id2token_map = {v: k for k, v in token2id_map.items()}

    def __getitem__(self, token):
        return self.token2id(token)

    def token2id(self, token):
        return self.token2id_map[token] if token in self.token2id_map else self.token2id_map['[UNK]']

    def id2token(self, id):
        return self.id2token_map[id]

    def __len__(self):
        return len(self.token2id_map)