convolution: conv2d_avx.c
	gcc -o $@ $^ ../preprocessing/im2col.c -lm -mavx2 -pthread

result1:
	./convolution /home/mirali/Conv2D/group2/1/input_tensor.bin /home/mirali/Conv2D/group2/1/kernel_tensor.bin FP32

result1_16:
	./convolution /home/mirali/Conv2D/group2/1/input_tensor.bin /home/mirali/Conv2D/group2/1/kernel_tensor.bin INT16

result1_32:
	./convolution /home/mirali/Conv2D/group2/1/input_tensor.bin /home/mirali/Conv2D/group2/1/kernel_tensor.bin INT32

result2:
	./convolution /home/mirali/Conv2D/group2/2/input_tensor.bin /home/mirali/Conv2D/group2/2/kernel_tensor.bin FP32

result2_16:
	./convolution /home/mirali/Conv2D/group2/2/input_tensor.bin /home/mirali/Conv2D/group2/2/kernel_tensor.bin INT16

result2_32:
	./convolution /home/mirali/Conv2D/group2/2/input_tensor.bin /home/mirali/Conv2D/group2/2/kernel_tensor.bin INT32

result3:
	./convolution /home/mirali/Conv2D/group2/3/input_tensor.bin /home/mirali/Conv2D/group2/3/kernel_tensor.bin FP32

result3_16:
	./convolution /home/mirali/Conv2D/group2/3/input_tensor.bin /home/mirali/Conv2D/group2/3/kernel_tensor.bin INT16

result3_32:
	./convolution /home/mirali/Conv2D/group2/3/input_tensor.bin /home/mirali/Conv2D/group2/3/kernel_tensor.bin INT32

clean:
	rm -f convolution
