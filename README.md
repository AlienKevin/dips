# DIPS: Efficient Multi-Criteria Cantonese Word Segmentation

We present DIPS, a novel approach to Cantonese word segmentation that addresses the challenges of balancing model
size, inference speed, and accuracy while accommodating diverse segmentation standards and supporting named
entity recognition. Our method combines fine-tuning, knowledge distillation, structured pruning, and
quantization to create a compact and efficient model. DIPS introduces a new segmentation scheme that captures
nuanced word boundaries, enabling flexible multi-criteria segmentation and entity identification.
Experimental results show that DIPS achieves comparable performance to ELECTRA Small on standard benchmarks
while being 17 times smaller (3 MB) and 3 times faster (0.13 ms/token on CPU). DIPS offers a practical solution
for Cantonese NLP tasks, particularly in resource-constrained environments and real-time applications. The model
is available as open-source libraries published on [PyPI](https://pypi.org/project/pydips/) and [NPM](https://www.npmjs.com/package/dips.js).

See my blog for details: https://kevinx.li/projects/dips

## Project Structure

### Segmentation Datasets

Use `segmentation_datasets/gen_dataset.py` to generate multi-tiered segmentations of the HKCanCor dataset.
The `data/` in `segmentation_datasets` is from [jacksonllee/multi-tiered-cantonese-word-segmentation](https://github.com/jacksonllee/multi-tiered-cantonese-word-segmentation). The result of running this script is already uploaded to HuggingFace at [AlienKevin/hkcancor-multi](https://huggingface.co/datasets/AlienKevin/hkcancor-multi).

After you trained ELECTRA Base model on this segmented dataset, you can distil it by generating more labels with the model on another larger dataset, like the Cantonese Wikipedia. Use `segmentation_datasets/distil_dataset.py` for the distillation. The result of running this script is already uploaded to HuggingFace at [AlienKevin/wiki-yue-long-multi](https://huggingface.co/datasets/AlienKevin/wiki-yue-long-multi).

### Training

Use `finetune-ckip-transformers/finetune_hkcancor_multi.py` to convert the segmented HKCanCor dataset to a suitable format for fine-tuning. The result of the script is stored at `finetune-ckip-transformers/data/finetune_hkcancor_multi.json`.

Use `finetune-ckip-transformers/finetune_multi.py` to convert both the segmented HKCanCor dataset and the distilled Cantonese Wikipedia dataset to a suitable format for fine-tuning. The result of the script is stored at `finetune-ckip-transformers/data/finetune_multi.json`.

To fine-tune the ELECTRA models for segmentation:
1. Install Hugging Face Transformers from [source](https://github.com/huggingface/transformers/).
2. Go to `transformers/examples/pytorch/token-classification/`.
3. Fine-tune ELECTRA Base model: `python run_ner.py --model_name_or_path toastynews/electra-hongkongese-base-discriminator --train_file finetune_hkcancor_multi.json --output_dir electra_base_hkcancor_multi --do_train`.
4. Distil the Base model by using it to label the Cantonese Wikipedia: `segmentation_datasets/distil_dataset.py`. Convert the dataset into suitable format for fine-tuning by running `finetune-ckip-transformers/finetune_multi.py`.
5. Then fine-tune an ELECTRA Small model on the distilled data: `python run_ner.py --model_name_or_path toastynews/electra-hongkongese-small-discriminator --train_file finetune_multi.json --output_dir electra_small_hkcancor_multi --do_train`. If you want to only keep the first n layers of the Small model, you can download it to a local folder and set `"num_hidden_layers": n` in the `config.json`. This way, `run_ner.py` will only load the first n layers during fine-tuning.
6. Lastly, you can truncate the vocabulary of the model by running `truncate_vocab.py`. It only keeps common Chinese characters, punctuations, and ASCII characters so the embedding can be drastically reduced. Feel free to modify the script to add/remove words.

Several fine-tuned models are available on HuggingFace:
1. ELECTRA Small fine-tuned on HKCanCor: [AlienKevin/electra-hongkongese-small-hkcancor-multi](https://huggingface.co/AlienKevin/electra-hongkongese-small-hkcancor-multi)
2. ELECTRA Base fine-tuned on HKCanCor: [AlienKevin/electra-hongkongese-base-hkcancor-multi](https://huggingface.co/AlienKevin/electra-hongkongese-base-hkcancor-multi). Best performing model but quite large and slow for deployment.
3. ELECTRA Small with top 6 layers dropped, fine-tuned on HKCanCor: [AlienKevin/electra-hongkongese-small-6-dropped-hkcancor-multi](https://huggingface.co/AlienKevin/electra-hongkongese-small-6-dropped-hkcancor-multi)
4. ELECTRA Small with top 6 layers dropped, fine-tuned on HKCanCor and Cantonese Wikipedia distilled from ELECTRA Base: [AlienKevin/electra-hongkongese-small-6-dropped-distilled-hkcancor-multi](https://huggingface.co/AlienKevin/electra-hongkongese-small-6-dropped-distilled-hkcancor-multi)
5. \[RECOMMANDED\] ELECTRA Small with top 6 layers dropped, fine-tuned on HKCanCor and Cantonese Wikipedia distilled from ELECTRA Base, with vocabulary truncated. [AlienKevin/electra-hongkongese-small-6-dropped-distilled-truncated-hkcancor-multi](https://huggingface.co/AlienKevin/electra-hongkongese-small-6-dropped-distilled-truncated-hkcancor-multi). This model achieves the best balance between size and performance. You can use the HuggingFace Transformers library to run this model for segmentation but we further quantized the model to be even smaller and faster. See the deployment section for details.

The `finetune-ckip-transformers` folder is derived from [toastynews/finetune-ckip-transformers](https://github.com/toastynews/finetune-ckip-transformers).

### Deployment

`bert.cpp/` includes a GGML runtime for our ELECTRA model. It also includes quantization functionality to further reduce the model size. See `bert.cpp/README.md` for detailed instructions to convert a fine-tuned Transformers model to GGUF, quantize the GGUF, and build the runtime for deployment to Mac, Linux, and WASM. Depending on the target platform, you will get different binaries. For macOS, you will get `build/src/libbert.dylib` and `ggml/src/libggml.dylib`. For Linux, you will get `build/src/libbert.so` and `ggml/src/libggml.so`. For WASM, you will get just one `build/bin/libbert.js` that contains WASM binaries inside. Refer to [AlienKevin/pydips](https://github.com/AlienKevin/pydips/tree/main) for how the macOS and Linux binaries may be embedded into a python library. Refer to [AlienKevin/dips.js](https://github.com/AlienKevin/dips.js) for how the WASM binary can be embedded into a JavaScript library.

The `bert.cpp` folder is derived from [iamlemec/bert.cpp](https://github.com/iamlemec/bert.cpp/). I added code to accomodate ELECTRA's embedding projection and the final token classifier layer at the end. I also added `src/wasm.cpp` as a WASM interface for GGML, inspired by [whisper.cpp](https://github.com/ggerganov/whisper.cpp/blob/master/examples/whisper.wasm/emscripten.cpp)
