# Quantized Conv2D layer implementation

## Table of Contents:
+ Data
  - group1 and gruop2: 
      - contains 3 different input and kernel tensors
+ Preprocessing
  - im2col and col2im implementations in __C__ for __cpu__ and __CUDA__ for __GPU__
+ Implementations
  - Problem 1. Naive implementation of convolution in C language.
  - Problem 2. Quantizing the naive implemetation of convolution using lower precisions.
  - Problem 3. Applying CPU vectorization using AVX instructions and Pthreads.
  - Problem 4. GPU vectorization using CUDA.
+ Report
  - Includes details and analysis of implemented algorithms and quantization errors
  
