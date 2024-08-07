import os
import json

pos_errors_path = 'outputs_v2/pos_errors.jsonl'

# Read the inputs from pos_errors.jsonl and filter out lines with specific error message
filtered_errors = []
if os.path.exists(pos_errors_path):
    with open(pos_errors_path, 'r', encoding='utf-8') as error_file:
        for line in error_file:
            error_result = json.loads(line)
            if "'error_msg': 'Insufficient Balance" not in error_result.get('error', ''):
                filtered_errors.append(line)

# Write the filtered errors back to pos_errors.jsonl
with open(pos_errors_path, 'w', encoding='utf-8') as error_file:
    for line in filtered_errors:
        error_file.write(line)
