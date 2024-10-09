# bert.cpp

This is a [ggml](https://github.com/ggerganov/ggml) implementation of the ELECTRA token classification model. It supports inference on CPU, CUDA and Metal in floating point and a wide variety of quantization schemes. Includes Python bindings for batched inference. The commands below were only tested on CPU.

### Install

Fetch this repository then download submodules and install packages with
```sh
git submodule update --init
pip install -r requirements.txt
```

To fetch models from HuggingFace and convert them to `gguf` format run something like the following
```sh
python bert_cpp/convert.py AlienKevin/electra-hongkongese-small-6-dropped-distilled-truncated-hkcancor-multi electra.gguf f32
```
This will convert to `float16` by default. To do `float32` add `f32` to the end of the command.

### Build

To build the C++ library for CPU/CUDA/Metal, run the following
```sh
# CPU
cmake -B build . && make -C build -j

# WASM
# Set the EMSCRIPTEN_WASM_ONLY option in CMakeLists.txt to ON to compile to a standalone WASM file
# Set to OFF to compile to a single JS file with glue code
emcmake cmake -B build . && make -C build -j

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
build/bin/main -m electra.gguf -p "阿張先生嗰時好nice㗎"

# Metal
GGML_METAL_PATH_RESOURCES=build/bin/ build/bin/main -m electra.gguf -p "阿張先生嗰時好nice㗎"
```
To force CPU usage, add the flag `-c`.

### Quantize

You can quantize models with the command (using the `f32` model as a base seems to work better)
```sh
build/bin/quantize electra.gguf electra-q8_0.gguf q8_0
```
or whatever your desired quantization level is. Currently supported values are: `q8_0`, `q5_0`, `q5_1`, `q4_0`, and `q4_1`. You can then pass these model files directly to `main` as above.
