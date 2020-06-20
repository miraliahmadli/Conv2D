convolution: conv2d_quantized.c
	gcc -o $@ $^ ../preprocessing/im2col.c -lm

result1_8:
	./convolution /home/mirali/Conv2D/group2/1/input_tensor.bin /home/mirali/Conv2D/group2/1/kernel_tensor.bin 8

result1_16:
	./convolution /home/mirali/Conv2D/group2/1/input_tensor.bin /home/mirali/Conv2D/group2/1/kernel_tensor.bin 16

result1_32:
	./convolution /home/mirali/Conv2D/group2/1/input_tensor.bin /home/mirali/Conv2D/group2/1/kernel_tensor.bin 32

result2_8:
	./convolution /home/mirali/Conv2D/group2/2/input_tensor.bin /home/mirali/Conv2D/group2/2/kernel_tensor.bin 8

result2_16:
	./convolution /home/mirali/Conv2D/group2/2/input_tensor.bin /home/mirali/Conv2D/group2/2/kernel_tensor.bin 16

result2_32:
	./convolution /home/mirali/Conv2D/group2/2/input_tensor.bin /home/mirali/Conv2D/group2/2/kernel_tensor.bin 32

result3_8:
	./convolution /home/mirali/Conv2D/group2/3/input_tensor.bin /home/mirali/Conv2D/group2/3/kernel_tensor.bin 8

result3_16:
	./convolution /home/mirali/Conv2D/group2/3/input_tensor.bin /home/mirali/Conv2D/group2/3/kernel_tensor.bin 16

result3_32:
	./convolution /home/mirali/Conv2D/group2/3/input_tensor.bin /home/mirali/Conv2D/group2/3/kernel_tensor.bin 32

clean:
	rm -f convolution
