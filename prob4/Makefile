convolution: conv2d_cuda.c
	nvcc -O3 -o $@ $^ ../preprocessing/im2col.c matmul.cu

result1:
	./convolution /home/mirali/Conv2D/group2/1/input_tensor.bin /home/mirali/Conv2D/group2/1/kernel_tensor.bin

result2:
	./convolution /home/mirali/Conv2D/group2/2/input_tensor.bin /home/mirali/Conv2D/group2/2/kernel_tensor.bin

result3:
	./convolution /home/mirali/Conv2D/group2/3/input_tensor.bin /home/mirali/Conv2D/group2/3/kernel_tensor.bin

clean:
	rm -f convolution
