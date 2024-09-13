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
# Collect data and group models
gguf_models = {'sizes': [], 'ud_yue_hk_f1': []}
electra_small_hkcancor_models = {'sizes': [], 'ud_yue_hk_f1': []}
electra_small_distilled_models = {'sizes': [], 'ud_yue_hk_f1': []}
other_models = {'names': [], 'sizes': [], 'ud_yue_hk_f1': []}

for model_name, model_data in results.items():
    model_size = get_model_size(model_name)
    if model_size is not None:
        if model_name.endswith('.gguf'):
            if model_name == 'electra.gguf':
                continue
            gguf_models['sizes'].append(model_size)
            gguf_models['ud_yue_hk_f1'].append(model_data['AlienKevin/ud_yue_hk']['token_f'])
        elif model_name.startswith('electra_small'):
            if 'hkcancor' in model_name:
                electra_small_hkcancor_models['sizes'].append(model_size)
                electra_small_hkcancor_models['ud_yue_hk_f1'].append(model_data['AlienKevin/ud_yue_hk']['token_f'])
            else:
                electra_small_distilled_models['sizes'].append(model_size)
                electra_small_distilled_models['ud_yue_hk_f1'].append(model_data['AlienKevin/ud_yue_hk']['token_f'])
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

# Create the plot
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
