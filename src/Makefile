# build all
all: conv_vanila conv_cpu conv_avx conv_gpu

conv_vanilla: ./conv2d.c
	gcc -O3 -o $@ $^ ../tools/im2col.c

conv_cpu: ./conv2d_quantized.c
	gcc -O3 -o $@ $^ ../tools/im2col.c -lm

conv_avx: ./conv2d_avx.c
	gcc -O3 -o $@ $^ ../tools/im2col.c -lm -mavx2 -pthread

conv_gpu: ./conv2d_cuda.c
	nvcc -O3 -o $@ $^ ../tools/im2col.c ../tools/matmul.cu

clean:
	rm -f *.out output_tensor.bin

# vanilla conv2d
vanilla_result_1:
	./conv_vanilla ../data/group2/1/input_tensor.bin ../data/group2/1/kernel_tensor.bin

vanilla_result_2:
	./conv_vanilla ../data/group2/2/input_tensor.bin ../data/group2/2/kernel_tensor.bin

vanilla_result_3:
	./conv_vanilla ../data/group2/3/input_tensor.bin ../data/group2/3/kernel_tensor.bin

vanilla_clean:
	rm -f conv_vanilla


# quantized conv2d
quantized_result_1_8:
	./conv_cpu ../data/group2/1/input_tensor.bin ../data/group2/1/kernel_tensor.bin 8

quantized_result_1_16:
	./conv_cpu ../data/group2/1/input_tensor.bin ../data/group2/1/kernel_tensor.bin 16

quantized_result_1_32:
	./conv_cpu ../data/group2/1/input_tensor.bin ../data/group2/1/kernel_tensor.bin 32

quantized_result_2_8:
	./conv_cpu ../data/group2/2/input_tensor.bin ../data/group2/2/kernel_tensor.bin 8

quantized_result_2_16:
	./conv_cpu ../data/group2/2/input_tensor.bin ../data/group2/2/kernel_tensor.bin 16

quantized_result_2_32:
	./conv_cpu ../data/group2/2/input_tensor.bin ../data/group2/2/kernel_tensor.bin 32

quantized_result_3_8:
	./conv_cpu ../data/group2/3/input_tensor.bin ../data/group2/3/kernel_tensor.bin 8

quantized_result_3_16:
	./conv_cpu ../data/group2/3/input_tensor.bin ../data/group2/3/kernel_tensor.bin 16

quantized_result_3_32:
	./conv_cpu ../data/group2/3/input_tensor.bin ../data/group2/3/kernel_tensor.bin 32

quantized_clean:
	rm -f conv_cpu

# avx conv2d
avx_result_1:
	./conv_avx ../data/group2/1/input_tensor.bin ../data/group2/1/kernel_tensor.bin FP32

avx_result_1_16:
	./conv_avx ../data/group2/1/input_tensor.bin ../data/group2/1/kernel_tensor.bin INT16

avx_result_1_32:
	./conv_avx ../data/group2/1/input_tensor.bin ../data/group2/1/kernel_tensor.bin INT32

avx_result_2:
	./conv_avx ../data/group2/2/input_tensor.bin ../data/group2/2/kernel_tensor.bin FP32

avx_result_2_16:
	./conv_avx ../data/group2/2/input_tensor.bin ../data/group2/2/kernel_tensor.bin INT16

avx_result_2_32:
	./conv_avx ../data/group2/2/input_tensor.bin ../data/group2/2/kernel_tensor.bin INT32

avx_result_3:
	./conv_avx ../data/group2/3/input_tensor.bin ../data/group2/3/kernel_tensor.bin FP32

avx_result_3_16:
	./conv_avx ../data/group2/3/input_tensor.bin ../data/group2/3/kernel_tensor.bin INT16

avx_result_3_32:
	./conv_avx ../data/group2/3/input_tensor.bin ../data/group2/3/kernel_tensor.bin INT32

avx_clean:
	rm -f conv_avx

# cuda conv
cuda_result_1:
	./conv_gpu ../data/group2/1/input_tensor.bin ../data/group2/1/kernel_tensor.bin FP32

cuda_result_1_16:
	./conv_gpu ../data/group2/1/input_tensor.bin ../data/group2/1/kernel_tensor.bin INT16

cuda_result_1_32:
	./conv_gpu ../data/group2/1/input_tensor.bin ../data/group2/1/kernel_tensor.bin INT32

cuda_result_2:
	./conv_gpu ../data/group2/2/input_tensor.bin ../data/group2/2/kernel_tensor.bin FP32

cuda_result_2_16:
	./conv_gpu ../data/group2/2/input_tensor.bin ../data/group2/2/kernel_tensor.bin INT16

cuda_result_2_32:
	./conv_gpu ../data/group2/2/input_tensor.bin ../data/group2/2/kernel_tensor.bin INT32

cuda_result_3:
	./conv_gpu ../data/group2/3/input_tensor.bin ../data/group2/3/kernel_tensor.bin FP32

cuda_result_3_16:
	./conv_gpu ../data/group2/3/input_tensor.bin ../data/group2/3/kernel_tensor.bin INT16

cuda_result_3_32:
	./conv_gpu ../data/group2/3/input_tensor.bin ../data/group2/3/kernel_tensor.bin INT32

cuda_clean:
	rm -f conv_gpu
