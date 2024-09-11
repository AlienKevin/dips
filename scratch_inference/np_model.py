import numpy as np


EMBEDDING_SIZE = 128
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 1024
NUM_ATTENTION_HEADS = 4
NUM_HIDDEN_LAYERS = 6
LAYER_NORM_EPS = 1e-12

class Linear:
    weight: np.ndarray
    bias: np.ndarray

    def __init__(self, weight: np.ndarray, bias: np.ndarray):
        self.weight = weight
        self.bias = bias
    
    def __call__(self, x):
        return x @ self.weight.T + self.bias


class LayerNorm():
    weight: np.ndarray
    bias: np.ndarray
    eps: float

    def __init__(self, weight: np.ndarray, bias: np.ndarray, eps: float=LAYER_NORM_EPS):
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def __call__(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x = (x - mean) / np.sqrt(variance + self.eps)  # normalize x to have mean=0 and var=1 over last axis
        return self.weight * x + self.bias  # scale and offset with gamma/beta params


class MultiHeadAttention:
    query: Linear
    key: Linear
    value: Linear
    output: Linear
    output_ln: LayerNorm

    def __call__(self, x):
        query_heads = np.split(self.query(x), NUM_ATTENTION_HEADS, axis=-1)
        key_heads = np.split(self.key(x), NUM_ATTENTION_HEADS, axis=-1)
        value_heads = np.split(self.value(x), NUM_ATTENTION_HEADS, axis=-1)

        attention_outputs = [
            softmax(q @ k.T / np.sqrt(q.shape[-1])) @ v
            for q, k, v in zip(query_heads, key_heads, value_heads)
        ]
        attention_outputs = np.concatenate(attention_outputs, axis=-1)
        attention_outputs = self.output(attention_outputs)
        attention_outputs = self.output_ln(attention_outputs + x)
        return attention_outputs


class EncoderLayer:
    attention: MultiHeadAttention
    intermediate: Linear # HIDDEN_SIZE, INTERMEDIATE_SIZE
    output: Linear # INTERMEDIATE_SIZE, HIDDEN_SIZE
    output_ln: LayerNorm

    def __call__(self, x):
        attention_output = self.attention(x)
        intermediate_output = gelu(self.intermediate(attention_output))
        layer_output = self.output(intermediate_output)
        layer_output = self.output_ln(layer_output + attention_output)
        return layer_output


class Electra:
    vocab: dict[str, int]
    word_embeddings: np.ndarray
    position_embeddings: np.ndarray
    token_type_embeddings: np.ndarray
    embeddings_layer_norm: LayerNorm
    embeddings_project: Linear
    encoder: list[EncoderLayer]
    classifier: Linear

    def __call__(self, x: list[int], token_type_ids: list[int]):
        embedded = self.word_embeddings[x] + self.position_embeddings[range(len(x))] + self.token_type_embeddings[token_type_ids]
        embedded = self.embeddings_layer_norm(embedded)
        embedded = self.embeddings_project(embedded)

        x = embedded

        for layer in self.encoder:
            x = layer(x)
        x = self.classifier(x)
        return x

    def tokenize(self, text: str):
        text = ''.join(text.split())
        tokens = list(text)
        return [self.vocab['[CLS]']] +\
            [self.vocab[token] if token in self.vocab else self.vocab['[UNK]'] for token in tokens] +\
            [self.vocab['[SEP]']]

    def cut(self, text: str):
        tokens = self.tokenize(text)
        token_type_ids = np.array([0] * len(tokens))
        out_logits = self(tokens, token_type_ids)[1:-1]
        predictions = np.argmax(out_logits, axis=-1)
        return list({'word': token, 'entity': 'DIPS'[pred]} for token, pred in zip(list(''.join(text.split())), predictions))


    def load(self, model_path):
        from pathlib import Path
        from safetensors import safe_open

        state_dict = {}

        with safe_open(Path(model_path)/"model.safetensors", framework="np") as f:
            for key in f.keys():
                state_dict[key.removeprefix('electra.')] = f.get_tensor(key)

        with open(Path(model_path)/"vocab.txt", "r") as f:
            vocab = f.read().splitlines()

        self.vocab = dict(enumerate(vocab))

        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"]
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"]
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"]
        self.embeddings_layer_norm = LayerNorm(
            weight=state_dict["embeddings.LayerNorm.weight"],
            bias=state_dict["embeddings.LayerNorm.bias"],
        )
        self.embeddings_project = Linear(
            weight=state_dict["embeddings_project.weight"],
            bias=state_dict["embeddings_project.bias"]
        )

        self.encoder = []

        for layer_index in range(NUM_HIDDEN_LAYERS):
            layer = EncoderLayer()
            layer.attention = MultiHeadAttention()
            layer.attention.query = Linear(
                weight=state_dict[f"encoder.layer.{layer_index}.attention.self.query.weight"],
                bias=state_dict[f"encoder.layer.{layer_index}.attention.self.query.bias"]
            )
            layer.attention.key = Linear(
                weight=state_dict[f"encoder.layer.{layer_index}.attention.self.key.weight"],
                bias=state_dict[f"encoder.layer.{layer_index}.attention.self.key.bias"]
            )
            layer.attention.value = Linear(
                weight=state_dict[f"encoder.layer.{layer_index}.attention.self.value.weight"],
                bias=state_dict[f"encoder.layer.{layer_index}.attention.self.value.bias"]
            )
            layer.attention.output = Linear(
                weight=state_dict[f"encoder.layer.{layer_index}.attention.output.dense.weight"],
                bias=state_dict[f"encoder.layer.{layer_index}.attention.output.dense.bias"]
            )
            layer.attention.output_ln = LayerNorm(
                weight=state_dict[f"encoder.layer.{layer_index}.attention.output.LayerNorm.weight"],
                bias=state_dict[f"encoder.layer.{layer_index}.attention.output.LayerNorm.bias"],
            )
            layer.intermediate = Linear(
                weight=state_dict[f"encoder.layer.{layer_index}.intermediate.dense.weight"],
                bias=state_dict[f"encoder.layer.{layer_index}.intermediate.dense.bias"]
            )
            layer.output = Linear(
                weight=state_dict[f"encoder.layer.{layer_index}.output.dense.weight"],
                bias=state_dict[f"encoder.layer.{layer_index}.output.dense.bias"]
            )
            layer.output_ln = LayerNorm(
                weight=state_dict[f"encoder.layer.{layer_index}.output.LayerNorm.weight"],
                bias=state_dict[f"encoder.layer.{layer_index}.output.LayerNorm.bias"],
            )
            self.classifier = Linear(
                weight=state_dict["classifier.weight"],
                bias=state_dict["classifier.bias"]
            )
            self.encoder.append(layer)


def gelu(x, approximate=True):
    if approximate:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    else:
        from scipy import special
        return 0.5 * x * (1 + special.erf(x / np.sqrt(2)))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


if __name__ == "__main__":
    model = Electra()
    model.load("../finetune-ckip-transformers/electra_small_layers_6_multi_compressed")
    x = np.array([33, 7427, 2989, 2850,   54,   51,   53,   47,  365,   36])
    token_type_ids = np.array([0] * len(x))
    out_logits = model(x, token_type_ids)

    reference_logits = np.array([[-0.2961, -0.4106, -0.4827,  1.0107],
         [-3.0762, -3.8008, -1.8193,  6.8945],
         [ 3.9219, -0.8179, -1.3418, -1.6943],
         [-2.2012,  0.3567,  3.9902, -1.5596],
         [-2.6641,  0.0366, -0.5991,  2.3164],
         [-1.3623,  4.6836, -1.8750, -1.9824],
         [-2.5215,  5.3789, -2.1699, -1.8330],
         [-1.5273,  4.2891, -1.5107, -1.9834],
         [-1.9482, -3.4570, -3.9316,  6.7969],
         [-0.2981, -0.4160, -0.4863,  1.0195]])

    # Calculate the average difference between out_logits and reference_logits
    avg_diff = np.mean(np.abs(out_logits - reference_logits))
    print(f"Average difference: {avg_diff}")
