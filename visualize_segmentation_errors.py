import json
import difflib

def visualize_segmentation_diff(reference, hypothesis):
    ref_string = ' '.join(reference)
    hyp_string = ' '.join(hypothesis)
    
    d = difflib.Differ()
    diff = list(d.compare(ref_string.split(), hyp_string.split()))
    
    result = []
    for item in diff:
        if item.startswith('  '):
            result.append(item[2:])
        elif item.startswith('- '):
            result.append(f"\033[91m{item[2:]}\033[0m")  # Red for deletions
        elif item.startswith('+ '):
            result.append(f"\033[92m{item[2:]}\033[0m")  # Green for additions
    
    return ' '.join(result)

def main():
    with open('errors.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            reference = [token for token, tag in data['reference']]
            hypothesis = [token for token, tag in data['hypothesis']]
            
            print('REF: ', ' '.join(reference))
            print('HYP: ', ' '.join(hypothesis))
            print('DIF: ', visualize_segmentation_diff(reference, hypothesis))
            print()

if __name__ == "__main__":
    main()

