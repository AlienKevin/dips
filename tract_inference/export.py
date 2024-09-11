from pathlib import Path
import transformers
from transformers.onnx import FeaturesManager
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import os

output_dir = "./model"
os.makedirs(output_dir, exist_ok=True)
model_path = Path(output_dir)/"model.onnx"

# load model and tokenizer
model_id = "../finetune-ckip-transformers/electra_small_layers_6_multi_compressed"
feature = "token-classification"
# Must specify torch_dtype=torch.float16 to load the fp16 model as fp16 in memory
model = AutoModelForTokenClassification.from_pretrained(model_id, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# load config
model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
onnx_config = model_onnx_config(model.config)

# export
onnx_inputs, onnx_outputs = transformers.onnx.export(
        preprocessor=tokenizer,
        model=model,
        config=onnx_config,
        opset=13,
        output=model_path,
)

tokenizer.save_pretrained(output_dir)
