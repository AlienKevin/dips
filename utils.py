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


def normalize(text):
    """
    Simplify traditional Chinese text to simplified Chinese.

    Args:
        text (str): The input traditional Chinese text.

    Returns:
        str: The simplified Chinese text.

    Example:
        >>> normalize("漢字")
        '汉字'
        >>> normalize("這是一個測試")
        '这是一个测试'
        >>> normalize("Hello, 世界!")
        'Hello, 世界!'
    """
    return ''.join(t2s.get(char, char) for char in text)


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