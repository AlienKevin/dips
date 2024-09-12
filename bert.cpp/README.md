# bert.cpp

**Note: Now that BERT support has been merged into [`llama.cpp`](https://github.com/ggerganov/llama.cpp), this repo is semi-defunct. The implementation in `llama.cpp` is substantially faster and has much better model support. Still happy to accept PRs if they do come along though.**

This is a [ggml](https://github.com/ggerganov/ggml) implementation of the BERT embedding architecture. It supports inference on CPU, CUDA and Metal in floating point and a wide variety of quantization schemes. Includes Python bindings for batched inference.

This repo is a fork of original [bert.cpp](https://github.com/skeskinen/bert.cpp) as well as [embeddings.cpp](https://github.com/xyzhang626/embeddings.cpp). Thanks to both of you!

### Install

Fetch this repository then download submodules and install packages with
```sh
git submodule update --init
pip install -r requirements.txt
```

To fetch models from `huggingface` and convert them to `gguf` format run something like the following (after creating the `models` directory)
```sh
python bert_cpp/convert.py BAAI/bge-base-en-v1.5 models/bge-base-en-v1.5-f16.gguf
```
This will convert to `float16` by default. To do `float32` add `f32` to the end of the command.

### Build

To build the C++ library for CPU/CUDA/Metal, run the following
```sh
# CPU
cmake -B build . && make -C build -j

# CUDA
cmake -DGGML_CUBLAS=ON -B build . && make -C build -j

# Metal
cmake -DGGML_METAL=ON -B build . && make -C build -j
```
On some distros, when compiling with CUDA, you also need to specify the host C++ compiler. To do this, I suggest setting the `CUDAHOSTCXX` environment variable to your C++ bindir.

### Execute

All executables are placed in `build/bin`. To run inference on a given text, run
```sh
# CPU / CUDA
build/bin/main -m models/bge-base-en-v1.5-f16.gguf -p "Hello world"

# Metal
GGML_METAL_PATH_RESOURCES=build/bin/ build/bin/main -m models/bge-base-en-v1.5-f16.gguf -p "Hello world"
```
To force CPU usage, add the flag `-c`.

### Python

You can also run everything through Python, which is particularly useful for batch inference. For instance,
```python
from bert_cpp import BertModel
mod = BertModel('models/bge-base-en-v1.5-f16.gguf')
emb = mod.embed(batch)
```
where `batch` is a list of strings and `emb` is a `numpy` array of embedding vectors.

### Quantize

You can quantize models with the command (using the `f32` model as a base seems to work better)
```sh
build/bin/quantize models/bge-base-en-v1.5-f32.gguf models/bge-base-en-v1.5-q8_0.gguf q8_0
```
or whatever your desired quantization level is. Currently supported values are: `q8_0`, `q5_0`, `q5_1`, `q4_0`, and `q4_1`. You can then pass these model files directly to `main` as above.
