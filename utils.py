import torch

# Read the STCharacters.txt file and create a mapping dictionary
t2s = {}
with open('data/STCharacters.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            simplified, traditional = parts
            for char in traditional.split():
                if char not in t2s:
                    t2s[char] = simplified


def halfen(s):
    '''
    Convert full-width characters to ASCII counterpart

    >>> halfen('１３')
    '13'
    >>> halfen('ＡａＺ')
    'AaZ'
    >>> halfen('（）【】“”「」！、，。：；')
    '()[]""""!,,.:;'
    '''
    FULL2HALF = dict((i + 0xFEE0, i) for i in range(0x21, 0x7F))
    FULL2HALF[0x3000] = 0x20
    return str(s).translate(FULL2HALF).replace('，', ',').replace('。', '.') \
        .replace('【', '[').replace('】', ']') \
        .replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'") \
        .replace('「', '"').replace('」', '"').replace('『', "'").replace('』', "'") \
        .replace('、', ',')


def simplify(text):
    """
    Simplify traditional Chinese text to simplified Chinese.

    Args:
        text (str): The input traditional Chinese text.

    Returns:
        str: The simplified Chinese text.

    Example:
        >>> simplify("漢字")
        '汉字'
        >>> simplify("這是一個測試")
        '这是一个测试'
        >>> simplify("Hello, 世界!")
        'Hello, 世界!'
    """
    return ''.join(t2s.get(char, char) for char in text)


def normalize(text):
    """
    Normalize text

    >>> normalize('（１３，漢：；字。）！')
    '(13,汉:;字.)!'
    """
    return halfen(simplify(text))


def pad_batch_seq(batch, max_sequence_length=None):
    if max_sequence_length is None:
        X = [item[0] for item in batch]
        y = [item[1] for item in batch]
    else:
        X = []
        y = []
        for item in batch:
            for i in range(0, len(item[0]), max_sequence_length):
                X.append(item[0][i:i+max_sequence_length])
                y.append(item[1][i:i+max_sequence_length])
    X_padded = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=-100)
    y_padded = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=-100)
    return X_padded, y_padded


def merge_tokens(tagged_characters, verbose=False):
    merged_tokens = []
    current_token = []
    current_tag = None

    for char, tag in tagged_characters:
        if tag.startswith('B-') or tag.startswith('S-'):
            if current_token:
                merged_tokens.append((''.join(current_token), current_tag))
            current_token = [char]
            current_tag = tag[2:]
        elif tag.startswith('I-') or tag.startswith('E-'):
            if current_tag is None:
                if verbose:
                    print(f"Error: I-tag '{tag}' without preceding B-tag. Treating as B-tag.")
                current_token = [char]
                current_tag = tag[2:]
            elif tag[2:] != current_tag:
                if verbose:
                    print(f"Error: I-tag '{tag}' does not match current B-tag '{current_tag}'. Overwriting with B-tag.")
                current_token.append(char)
            else:
                current_token.append(char)
        else:
            if current_token:
                merged_tokens.append((''.join(current_token), current_tag))
                current_token = []
                current_tag = None
            merged_tokens.append((char, tag))

    if current_token:
        merged_tokens.append((''.join(current_token), current_tag))

    return merged_tokens


def fix_tag(tag):
    if tag == 'V':
        return 'VERB'
    elif tag == '[PAD]':
        return 'X'
    return tag


def score_tags(test_dataset, tag):
    from tqdm import tqdm
    from spacy.training import Example
    from spacy.scorer import Scorer
    from spacy.tokens import Doc
    from spacy.vocab import Vocab
    
    V = Vocab()
    examples = []
    errors = []
    for reference in tqdm(test_dataset):
        hypothesis = merge_tokens(tag(''.join(token for token, _ in reference)))
        reference_tokens = [token for token, _ in reference]
        target = Doc(V, words=reference_tokens, spaces=[False for _ in reference], pos=[fix_tag(tag) for _, tag in reference])
        predicted_doc = Doc(V, words=[token for token, _ in hypothesis], spaces=[False for _ in hypothesis], pos=[fix_tag(tag) for _, tag in hypothesis])
        example = Example(predicted_doc, target)
        examples.append(example)

        if reference != hypothesis:
            errors.append({'reference': reference, 'hypothesis': hypothesis})

    scorer = Scorer()
    results = scorer.score(examples)

    return results, errors


def read_pretrained_embeddings(embedding_path, vocab, freeze=True):
    word_to_embed = {}
    unknown_embeds = []
    with open(embedding_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            split = line.rstrip().split(' ')
            if len(split) > 2:
                word = split[0]
                vec = torch.tensor([float(x) for x in split[1:]])
                if word in vocab:
                    word_to_embed[word] = vec
                else:
                    unknown_embeds.append(vec)
    embedding_dim = next(iter(word_to_embed.values())).size(0)
    out = torch.empty(len(vocab), embedding_dim)
    torch.nn.init.uniform_(out, -0.8, 0.8)
    
    for word, embed in word_to_embed.items():
        out[vocab[word]] = embed
    
    if unknown_embeds:
        unk_embed = torch.stack(unknown_embeds).mean(dim=0)
        out[vocab['[UNK]']] = unk_embed
    
    return torch.nn.Embedding.from_pretrained(out, freeze=freeze)
