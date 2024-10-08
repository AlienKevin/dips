import json
import matplotlib.pyplot as plt
import os

# Read multi_model_results.json
with open('multi_model_results.json', 'r') as f:
    results = json.load(f)

# Initialize lists to store data
model_names = []
model_sizes = []
hkcancor_accuracies = []
ud_yue_hk_f1 = []
ud_zh_hk_f1 = []
cityu_seg_f1 = []

# Function to get model size
def get_model_size(model_name):
    if model_name.endswith('.gguf'):
        path = f'bert.cpp/{model_name}'
    else:
        path = f'finetune-ckip-transformers/{model_name}/model.safetensors'
    
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)  # Convert to MB
    else:
        print(f"Warning: File not found for {model_name}")
        return None

# Function to calculate latency
def get_latency(model_data):
    return model_data['total_time'] / model_data['total_tokens']

# Collect data and group models
gguf_models = {'sizes': [], 'ud_yue_hk_f1': [], 'ud_zh_hk_f1': [], 'latencies': [], 'hkcancor_accuracies': []}
electra_small_hkcancor_models = {'sizes': [], 'ud_yue_hk_f1': [], 'ud_zh_hk_f1': [], 'latencies': [], 'hkcancor_accuracies': []}
electra_small_distilled_models = {'sizes': [], 'ud_yue_hk_f1': [], 'ud_zh_hk_f1': [], 'latencies': [], 'hkcancor_accuracies': []}
other_models = {'names': [], 'sizes': [], 'ud_yue_hk_f1': [], 'ud_zh_hk_f1': [], 'latencies': [], 'hkcancor_accuracies': []}

for model_name, model_data in results.items():
    model_size = get_model_size(model_name)
    latency = get_latency(model_data)
    if model_size is not None:
        if model_name.endswith('.gguf'):
            if model_name == 'electra.gguf' or model_name == 'electra-q4_1.gguf':
                continue
            gguf_models['sizes'].append(model_size)
            gguf_models['ud_yue_hk_f1'].append(model_data['AlienKevin/ud_yue_hk']['token_f'])
            gguf_models['ud_zh_hk_f1'].append(model_data['AlienKevin/ud_zh_hk']['token_f'])
            gguf_models['latencies'].append(latency)
            gguf_models['hkcancor_accuracies'].append(model_data['AlienKevin/hkcancor-multi']['accuracy'])
        elif model_name.startswith('electra_small'):
            if 'hkcancor' in model_name:
                electra_small_hkcancor_models['sizes'].append(model_size)
                electra_small_hkcancor_models['ud_yue_hk_f1'].append(model_data['AlienKevin/ud_yue_hk']['token_f'])
                electra_small_hkcancor_models['ud_zh_hk_f1'].append(model_data['AlienKevin/ud_zh_hk']['token_f'])
                electra_small_hkcancor_models['latencies'].append(latency)
                electra_small_hkcancor_models['hkcancor_accuracies'].append(model_data['AlienKevin/hkcancor-multi']['accuracy'])
            else:
                electra_small_distilled_models['sizes'].append(model_size)
                electra_small_distilled_models['ud_yue_hk_f1'].append(model_data['AlienKevin/ud_yue_hk']['token_f'])
                electra_small_distilled_models['ud_zh_hk_f1'].append(model_data['AlienKevin/ud_zh_hk']['token_f'])
                electra_small_distilled_models['latencies'].append(latency)
                electra_small_distilled_models['hkcancor_accuracies'].append(model_data['AlienKevin/hkcancor-multi']['accuracy'])
        else:
            if model_name == 'electra_base_hkcancor_multi':
                display_name = 'Base'
            elif model_name == 'electra_large_hkcancor_multi':
                display_name = 'Large'
            elif model_name == 'albert_tiny_chinese_hkcancor_multi':
                continue
            elif model_name == 'bert_tiny_chinese_hkcancor_multi':
                continue
            else:
                display_name = model_name
            other_models['names'].append(display_name)
            other_models['sizes'].append(model_size)
            other_models['ud_yue_hk_f1'].append(model_data['AlienKevin/ud_yue_hk']['token_f'])
            other_models['ud_zh_hk_f1'].append(model_data['AlienKevin/ud_zh_hk']['token_f'])
            other_models['latencies'].append(latency)
            other_models['hkcancor_accuracies'].append(model_data['AlienKevin/hkcancor-multi']['accuracy'])

# Create the plot for performance vs size
plt.figure(figsize=(6, 4))

colors = ['#e41a1c', '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#dede00']

# Plot GGUF models as a line
plt.plot(gguf_models['sizes'], gguf_models['ud_yue_hk_f1'], 'o-', color=colors[0], label='Ours')

# Plot ELECTRA Small models as a line
plt.plot(electra_small_hkcancor_models['sizes'], electra_small_hkcancor_models['ud_yue_hk_f1'], 's-', color=colors[-1], label='Small (Layer Dropped)')

# Plot ELECTRA Small Distilled models as a line
plt.plot(electra_small_distilled_models['sizes'], electra_small_distilled_models['ud_yue_hk_f1'], '^-', color=colors[-2], label='Small (Distilled)')

# Plot other models with different shapes and colors
markers = ['D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
for i, (name, size, f1) in enumerate(zip(other_models['names'], other_models['sizes'], other_models['ud_yue_hk_f1'])):
    plt.scatter(size, f1, marker=markers[i % len(markers)], c=colors[1:-2][i % len(colors)], label=name)
plt.xlabel('Model Size (MB)')
plt.ylabel('F1 Score')
plt.xscale('log')  # Use log scale for x-axis due to potentially large size differences
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.gcf().set_size_inches(7.5, 4)  # Increase width from 6 to 10 inches, keep height at 4 inches

# Move legend to the top and reorganize
handles, labels = plt.gca().get_legend_handles_labels()
order = ['Ours', 'Small (Distilled)', 'Small (Layer Dropped)', 'Base', 'Large']
handles_dict = dict(zip(labels, handles))
ordered_handles = [handles_dict[label] for label in order if label in handles_dict]
ordered_labels = [label for label in order if label in handles_dict]

plt.legend(ordered_handles, ordered_labels, bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=5, frameon=False)

plt.tight_layout()
plt.savefig('multi_model_performance_vs_size.png', dpi=300, bbox_inches='tight')
plt.close()

# Create the plot for performance vs latency
plt.figure(figsize=(6, 4))

# Plot GGUF models as a line
plt.plot(gguf_models['latencies'], gguf_models['ud_yue_hk_f1'], 'o-', color=colors[0], label='Ours')

# Plot ELECTRA Small models as a line
plt.plot(electra_small_hkcancor_models['latencies'], electra_small_hkcancor_models['ud_yue_hk_f1'], 's-', color=colors[-1], label='Small (Layer Dropped)')

# Plot ELECTRA Small Distilled models as a line
plt.plot(electra_small_distilled_models['latencies'], electra_small_distilled_models['ud_yue_hk_f1'], '^-', color=colors[-2], label='Small (Distilled)')

# Plot other models with different shapes and colors
for i, (name, latency, f1) in enumerate(zip(other_models['names'], other_models['latencies'], other_models['ud_yue_hk_f1'])):
    plt.scatter(latency, f1, marker=markers[i % len(markers)], c=colors[1:-2][i % len(colors)], label=name)

plt.xlabel('Latency (seconds/token)')
plt.ylabel('F1 Score')
plt.xscale('log')  # Use log scale for x-axis due to potentially large latency differences
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.gcf().set_size_inches(7.5, 4)

# Move legend to the top and reorganize
handles, labels = plt.gca().get_legend_handles_labels()
order = ['Ours', 'Small (Distilled)', 'Small (Layer Dropped)', 'Base', 'Large']
handles_dict = dict(zip(labels, handles))
ordered_handles = [handles_dict[label] for label in order if label in handles_dict]
ordered_labels = [label for label in order if label in handles_dict]

plt.legend(ordered_handles, ordered_labels, bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=5, frameon=False)

plt.tight_layout()
plt.savefig('multi_model_performance_vs_latency.png', dpi=300, bbox_inches='tight')
plt.close()

# Create the plot for HKCanCor accuracy vs size
plt.figure(figsize=(6, 4))

# Plot GGUF models as a line
plt.plot(gguf_models['sizes'], gguf_models['hkcancor_accuracies'], 'o-', color=colors[0], label='Ours')

# Plot ELECTRA Small models as a line
plt.plot(electra_small_hkcancor_models['sizes'], electra_small_hkcancor_models['hkcancor_accuracies'], 's-', color=colors[-1], label='Small (Layer Dropped)')

# Plot ELECTRA Small Distilled models as a line
plt.plot(electra_small_distilled_models['sizes'], electra_small_distilled_models['hkcancor_accuracies'], '^-', color=colors[-2], label='Small (Distilled)')

# Plot other models with different shapes and colors
for i, (name, size, accuracy) in enumerate(zip(other_models['names'], other_models['sizes'], other_models['hkcancor_accuracies'])):
    plt.scatter(size, accuracy, marker=markers[i % len(markers)], c=colors[1:-2][i % len(colors)], label=name)

plt.xlabel('Model Size (MB)')
plt.ylabel('Accuracy')
plt.xscale('log')  # Use log scale for x-axis due to potentially large size differences
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.gcf().set_size_inches(7.5, 4)

# Move legend to the top and reorganize
handles, labels = plt.gca().get_legend_handles_labels()
order = ['Ours', 'Small (Distilled)', 'Small (Layer Dropped)', 'Base', 'Large']
handles_dict = dict(zip(labels, handles))
ordered_handles = [handles_dict[label] for label in order if label in handles_dict]
ordered_labels = [label for label in order if label in handles_dict]

plt.legend(ordered_handles, ordered_labels, bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=5, frameon=False)

plt.tight_layout()
plt.savefig('multi_model_hkcancor_accuracy_vs_size.png', dpi=300, bbox_inches='tight')
plt.close()


# Create a Markdown table with model information
print("| Model | Size (MB) | Latency (ms/token) | UD Yue F1 | UD Zh F1 |")
print("|-------|-----------|---------------------|-----------|----------|")

# Function to format float values
def format_float(value):
    return f"{value:.4f}"

# GGUF models
for size, latency, ud_yue_f1, ud_zh_f1 in zip(gguf_models['sizes'], gguf_models['latencies'], gguf_models['ud_yue_hk_f1'], gguf_models['ud_zh_hk_f1']):
    print(f"| Ours | {size:.2f} | {latency*1000:.2f} | {format_float(ud_yue_f1)} | {format_float(ud_zh_f1)} |")

# ELECTRA Small (Layer Dropped) models
for size, latency, ud_yue_f1, ud_zh_f1 in zip(electra_small_hkcancor_models['sizes'], electra_small_hkcancor_models['latencies'], electra_small_hkcancor_models['ud_yue_hk_f1'], electra_small_hkcancor_models['ud_zh_hk_f1']):
    print(f"| Small (Layer Dropped) | {size:.2f} | {latency*1000:.2f} | {format_float(ud_yue_f1)} | {format_float(ud_zh_f1)} |")

# ELECTRA Small (Distilled) models
for size, latency, ud_yue_f1, ud_zh_f1 in zip(electra_small_distilled_models['sizes'], electra_small_distilled_models['latencies'], electra_small_distilled_models['ud_yue_hk_f1'], electra_small_distilled_models['ud_zh_hk_f1']):
    print(f"| Small (Distilled) | {size:.2f} | {latency*1000:.2f} | {format_float(ud_yue_f1)} | {format_float(ud_zh_f1)} |")

# Other models
for name, size, latency, ud_yue_f1, ud_zh_f1 in zip(other_models['names'], other_models['sizes'], other_models['latencies'], other_models['ud_yue_hk_f1'], other_models['ud_zh_hk_f1']):
    print(f"| {name} | {size:.2f} | {latency*1000:.2f} | {format_float(ud_yue_f1)} | {format_float(ud_zh_f1)} |")
