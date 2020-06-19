convolution: conv2d.c
	gcc -o $@ $^ preprocessing/im2col.c

result1: convolution
	./convolution ./group2/1/input_tensor.bin ./group2/1/kernel_tensor.bin

result2: convolution
	./convolution ./group2/2/input_tensor.bin ./group2/2/kernel_tensor.bin

result3: convolution
	./convolution ./group2/3/input_tensor.bin ./group2/3/kernel_tensor.bin

clean:
	rm -f convolution
