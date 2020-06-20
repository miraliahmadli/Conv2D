/*

For inputs, the 16 bytes are for (N, H, W, C), where N is the batch size, H is the height, W is
the width, and C is the channel. 

For kernels, the 16 bytes are for (KH, KW, OC, IC), where
KH is the kernel height, KW is the kernel width, OC is the output channels, and IC is the input
channels. All of these values are integer. 

The following N*H*W*C*4 bytes will contain the single-precision floating point (FP32) numerals, 
which are the values in the 1-D flattened array of input tensor. 

Assume the padding mode is always the “SAME”, which means the shape of
input tensor is maintained in the output tensor. 

Assume the stride is always 1. Use solely the primitive scalar operations (i.e., +, -, *, /).

*/

#include <stdio.h>
#include <stdlib.h>
#include "assert.h"
#include <time.h>

#include "../preprocessing/im2col.h"
#include "matmul.h"

float *input, *kernel, *output;
float *input_col, *output_col;
int N, H, W, C;
int KH, KW, OC, IC;


//  reads binary data from given files to input and kernel arrays
void read_data(const char *input_file, const char *kernel_file){
    FILE *in, *kern;
    in = fopen(input_file, "rb");
    kern = fopen(kernel_file, "rb");

    if (in == NULL || kern == NULL){
        printf("Could not open the input bin files %s %s\n", input_file, kernel_file);
        if (in == NULL) printf("input bin is NULL\n");
        if (kern == NULL) printf("kernel bin is NULL\n");
        fclose(in);
        fclose(kern);
        exit(-1);
    }

    size_t input_read = 0;
    size_t kernel_read = 0;

    //  read dimensions
    input_read += fread(&N, 4, 1, in);
    input_read += fread(&H, 4, 1, in);
    input_read += fread(&W, 4, 1, in);
    input_read += fread(&C, 4, 1, in);
    if (input_read != 4){
        printf("Could not read dimensions, read input elems = %ld\n", input_read);
        fclose(in);
        fclose(kern);
        exit(3);
    }
    // printf("N = %d H = %d W = %d C = %d\n", N, H, W, C);

    kernel_read += fread(&KH, 4, 1, kern);
    kernel_read += fread(&KW, 4, 1, kern);
    kernel_read += fread(&OC, 4, 1, kern);
    kernel_read += fread(&IC, 4, 1, kern);
    if (kernel_read != 4){
        printf("Could not read dimensions, read kernel elems = %ld\n", kernel_read);
        fclose(in);
        fclose(kern);
        exit(3);
    }
    // printf("KH = %d KW = %d OC = %d IC = %d\n", KH, KW, OC, IC);

    //  allocate arrays
    float *input_pre = (float *)malloc(N*H*W*C*4);
    // input = (float *)malloc(N*H*W*C*4);
    //  A is (KH*KW*C) * (N*H*W) matrix
    input_col = (float *)malloc((KH*KW*C) * (N*H*W) * 4);
    int pad = (KH - 1)/2;

    //  read input image
    // input_read += fread(input, 4, N*H*W*C, in);
    input_read += fread(input_pre, 4, N*H*W*C, in);
    if (input_read != N*H*W*C + 4){
        printf("Could not write dimensions, written elems = %ld\n", input_read);
        fclose(in);
        fclose(kern);
        free(input);
        free(kernel);
        exit(3);
    }

    //  input = input_pre.transpose(n, c, h, w)
    input = (float *)malloc(N*H*W*C*4);
    for(int n = 0; n < N; n++){
        for(int h = 0; h < H; h++){
            for(int w = 0; w < W; w++){
                for(int c = 0; c < C; c++){
                    input[n*H*W*C + c*H*W + h*W + w] = input_pre[n*H*W*C + h*W*C + w*C + c];
                }
            }
        }
    }
    free(input_pre);

    // apply im2col algorithm
    im2col_cpu(input, C, H, W, KH, 1, pad, input_col);
    free(input);

    // read filters
    // kernel = (float *)malloc(KH*KW*OC*IC*4);
    float *kernel_pre = (float *)malloc(KH*KW*OC*IC*4);
    kernel_read += fread(kernel_pre, 4, KH*KW*OC*IC, kern);
    if (kernel_read != KH*KW*OC*IC + 4){
        printf("Could not write dimensions, written elems = %ld\n", kernel_read);
        fclose(in);
        fclose(kern);
        free(input);
        free(kernel_pre);
        exit(3);
    }

    //  kernel = kernel.transpose(3, 0, 1, 2)
    kernel = (float *)malloc(KH*KW*OC*IC*4);
    for(int kh = 0; kh < KH; kh++){
        for(int kw = 0; kw < KW; kw++){
            for(int oc = 0; oc < OC; oc++){
                for(int ic = 0; ic < IC; ic++){
                    kernel[oc*KH*KW*C + ic*KH*KW + kh*KW + kw] = kernel_pre[kh*KW*OC*IC + kw*OC*IC + ic*OC + oc];
                }
            }
        }
    }
    free(kernel_pre);

    //  close files
    fclose(in);
    fclose(kern);
}


//  writes the results from output to binary file
void write_data(const char *output_file){
    FILE *out;
    out = fopen(output_file, "wb");
    if (out == NULL){
        printf("Could not open the output bin file\n");
        exit(-1);
    }

    //  for debugging
    size_t elems_written = 0;
    
    //  write output dimensions
    elems_written += fwrite(&N, 4, 1, out);
    elems_written += fwrite(&H, 4, 1, out);
    elems_written += fwrite(&W, 4, 1, out);
    elems_written += fwrite(&OC, 4, 1, out);
    if (elems_written != 4){
        printf("Could not write dimensions, written elems = %ld\n", elems_written);
        fclose(out);
        exit(2);
    }
    // printf("N = %d H = %d W = %d OC = %d\n", N, H, W, OC);

    //  write output matrix
    elems_written += fwrite(output, 4, N*H*W*OC, out);
    if (elems_written != N*H*W*OC + 4){
        printf("Could not write dimensions, written elems = %ld\n", elems_written);
        fclose(out);
        exit(2);
    }

    fclose(out);
    free(output);
}

//  convolution operation
//  apply kernel on input and save results on output
double conv2d(){
    //  multiplication of kernel and input image
    //  kernel dimension: (OC) * (KH * KW * C)
    //  A dimension: (KH * KW * C) * (N * H * W)
    //  output dimension: OC * N * H * W (or maybe OC * H * W * N)
    //  we need to convert it to N * H * W * C (by transposing)

    int Y = KH * KW * C;
    int Z = N * H * W;
    int X = OC;

    output_col = (float *) malloc(X * Y * sizeof(float));
    clock_t start, end;
    double cpu_time_used;
    
    //  Convolution operation
    start = clock();
    matmul(output_col, kernel, input_col, X, Y, Z);
    end = clock();

    free(input_col);
    free(kernel);

    int pad = (KH - 1)/2;
    output = (float *) malloc(X * Y * sizeof(float));
    
    //  output = output_col.transpose(3, 1, 2, 0)
    //  outpu_col dimension: OC * H * W * N
    //  output will have dimension: N * H * W * OC
    for(int oc = 0; oc < OC; oc++){
        for(int h = 0; h < H; h++){
            for(int w = 0; w < W; w++){
                for(int n = 0; n < N; n++){
                    output[n*H*W*OC + h*W*OC + w*OC + oc] = output_col[oc*Z + h*W*N + w*N + n];
                }
            }
        }
    }
    free(output_col);

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    return cpu_time_used;
}


int main(int argc, char *argv[]){
    //  read data from binary files
    read_data(argv[1], argv[2]);
    //  preprocessing (this part can be handled in reading part)

    //  convolution operation
    //  kernel will be applied to input and result will be stored in output
    double elapsed_time = conv2d();
    // printf("Time took for convolution operation is: %f\n", elapsed_time);
    printf("Convolution operation took %f seconds to execute\n", elapsed_time); 
    //  write results to binary file
    write_data("output_tensor.bin");

    return 0;
}
