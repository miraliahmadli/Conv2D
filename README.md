# Quantized Conv2D layer implementation

## About
This is an implementation of convolution operation using following optimization strategies:
- **Naive quantization.**
Linear quantization using uniform scale value.
- **CPU parallelization.**
Parallelization using multithreading (pthread) and AVX instructions.
- **GPU parallelization.**
Parallelization using CUDA.

## Table of Contents:
+ Data
  - group1 and gruop2: 
      - Contains 3 different input and kernel tensors
+ Preprocessing
  - im2col and col2im implementations in __C__ for __cpu__ and __CUDA__ for __GPU__
  - matrix multiplication implementations in __CUDA__ for __GPU__
+ Implementations
  - `./src/conv2d.c`: Naive implementation of convolution in C language.
  - `./src/conv2d_quantized.c`: Quantizing the naive implemetation of convolution using lower precisions.
  - `./src/conv2d_avx.c`: Applying CPU vectorization using AVX instructions and Pthreads.
  - `./src/conv2d_cuda.c`: GPU vectorization using CUDA.
+ Report
  - Includes details and analysis of implemented algorithms and quantization errors
  
## Usage

### Input / Output file format
`conv2d_*` programs take 2 binary files as input.  
- **input tensor.** First 16 bytes are for `(N, H, W, IC)`, where `N` is the batch size, `H` is the height, `W` is the width, and `IC` is the channel.
- **kernel tensor.** First 16 bytes are for `(KH, KW, OC, IC)`, where `KH` is the kernel height, `KW` is the kernel width, `OC` is the output channel, and `IC` is the input channel.  

They produce 1 binary file as output.
- **output tensor.** First 16 bytes are for `(N, H, W, OC)`, where `N` is the batch size, `H` is the height, `W` is the width, and `OC` is the channel.

For all binary files, following bytes after first 16 bytes are the real tensor data, which follows the memory order corresponding to the dimension rule written above.

### Running command
At `src/` directory,
```
$ make
```

**conv_vanilla**
```
$ ./conv_vanila $(INPUT_BIN_PATH) $(OUTPUT_BIN_PATH)
```

**conv_cpu**
```
$ ./conv_cpu $(INPUT_BIN_PATH) $(OUTPUT_BIN_PATH) [8,16,32]
```
Third argument is mandatory and indicates the level of precision.

**conv_avx**
```
$ ./conv_avx $(INPUT_BIN_PATH) $(OUTPUT_BIN_PATH) [FP32/INT32/INT16]
```
Third argument is mandatory. For `FP32`, no quantization is applied. For `INT*`, quantization using integer of corresponding number of bits is applied.

**conv_gpu**
```
$ ./conv_gpu $(INPUT_BIN_PATH) $(OUTPUT_BIN_PATH) [FP32/INT32/INT16]
```
Same as **conv_avx**

Output will be Normalized Mean Square Error obtained from quantization operation and **`output_tensor.bin`**
