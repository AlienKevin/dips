from collections import defaultdict
import regex

def bpe(tokens, vocab_size):
    # i. Initialize vocab
    vocab = set()
    tokens_next = []
    for token in tokens:
        token_next = []
        for c in token:
            vocab.add(c)
            token_next.append(c)
        tokens_next.append(token_next)
    tokens = tokens_next
    merge_rules = []

    # Calculate character pair frequencies
    freqs = defaultdict(int)
    for token in tokens:
        for i in range(1, len(token)):
            freqs[(token[i-1], token[i])] += 1

    while len(vocab) < vocab_size:
        # Merge the most common pair
        most_freq_pair = None
        highest_freq = 0
        for pair, freq in freqs.items():
            if freq > highest_freq:
                most_freq_pair = pair
                highest_freq = freq
        vocab.add(''.join(most_freq_pair))
        merge_rules.append(most_freq_pair)
        # Update corpus by merging most frequent pair
        for token in tokens:
            i = 1
            while i < len(token):
                if (token[i-1], token[i]) == most_freq_pair:
                    freqs[(token[i-1], token[i])] = 0
                    if i - 2 >= 0:
                        freqs[(token[i-2], token[i-1])] -= 1
                    if i + 1 < len(token):
                        freqs[(token[i], token[i+1])] -= 1
                    token[i-1] += token[i]
                    token.pop(i)
                    if i - 2 >= 0:
                        freqs[(token[i-2], token[i-1])] += 1
                    if i < len(token):
                        freqs[(token[i-1], token[i])] += 1
                i += 1
    return (list(sorted(vocab)), merge_rules, tokens)

def plot_stats(tokens):
    import matplotlib.pyplot as plt
    import numpy as np
    lengths = [len(token) for token in tokens]
    max_length = max(lengths)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Histogram
    ax1.hist(lengths, bins=range(1, max_length + 2), align='left', rwidth=0.8)
    ax1.set_xlabel('Length')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Token Lengths')
    ax1.set_xticks(range(1, max_length + 1))
    
    # IQR plot
    q1, median, q3 = np.percentile(lengths, [25, 50, 75])
    iqr = q3 - q1
    lower_bound = max(1, q1 - 1.5 * iqr)
    upper_bound = q3 + 1.5 * iqr
    
    ax2.boxplot(lengths, vert=False, whis=[0, 100])
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel('Length')
    ax2.set_yticks([])
    
    # Add IQR annotations
    ax2.annotate(f'Q1: {q1:.2f}', xy=(q1, 1), xytext=(q1, 1.1), ha='center', va='bottom', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax2.annotate(f'Median: {median:.2f}', xy=(median, 1), xytext=(median, 1.2), ha='center', va='bottom', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax2.annotate(f'Q3: {q3:.2f}', xy=(q3, 1), xytext=(q3, 1.1), ha='center', va='bottom', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax2.annotate(f'IQR: {iqr:.2f}', xy=((q1+q3)/2, 1), xytext=((q1+q3)/2, 1.3), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    table = {}

    with open('data/Cangjie5_SC.txt', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split('\t')
            code = list(parts[1])
            # Mark the last component as ending a Chinese character
            code[-1] = code[-1].upper()
            # Skip punctuations
            if regex.match(r'^\p{Punct}$', parts[0]):
                continue
            # Use the first code for each character
            if parts[0] not in table:
                table[parts[0]] = code

    plot_stats(table.values())

    vocab, rules, tokens = bpe(list(table.values()), 400)

    plot_stats(tokens)

    # Export BPE token table to Cangjie5_SC_BPE.txt
    with open('data/Cangjie5_SC_BPE.txt', 'w', encoding='utf-8') as f:
        for char, code in zip(table.keys(), tokens):
            # Convert BPE tokens to their corresponding indices
            code_indices = ' '.join(token for token in code)
            # Write the character and its encoded BPE tokens to the file
            f.write(f"{char}\t{code_indices}\n")

    print("BPE token table exported to data/Cangjie5_SC_BPE.txt")
