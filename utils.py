import torch

# Read the STCharacters.txt file and create a mapping dictionary
t2s = {}
with open('data/STCharacters.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            simplified, traditional = parts
            for char in traditional.split():
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


def pad_batch_seq(batch, padding_value, max_sequence_length=None):
    if max_sequence_length is None:
        X = [item[0] for item in batch]
        y = [item[1] for item in batch]
    else:
        X = [item[0][:max_sequence_length] for item in batch]
        y = [item[1][:max_sequence_length] for item in batch]
    X_padded = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=padding_value)
    y_padded = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=padding_value)
    return X_padded, y_padded