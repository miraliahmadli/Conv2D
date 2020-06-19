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

#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include "assert.h"

#include "preprocessing/im2col.h"
#include "preprocessing/col2im.h"
using namespace std;

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
        printf("Could not open the input bin files");
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
        printf("Could not read dimensions, read input elems = %d", input_read);
        fclose(in);
        fclose(kern);
        exit(3);
    }

    kernel_read += fread(&KH, 4, 1, in);
    kernel_read += fread(&KW, 4, 1, in);
    kernel_read += fread(&OC, 4, 1, in);
    kernel_read += fread(&IC, 4, 1, in);
    if (kernel_read != 4){
        printf("Could not read dimensions, read kernel elems = %d", kernel_read);
        fclose(in);
        fclose(kern);
        exit(3);
    }

    //  allocate arrays
    input = (float *)malloc(N*H*W*C*4);
    //  A is (KH*KW*C) * (N*H*W) matrix
    input_col = (float *)malloc((KH*KW*C) * (N*H*W) * 4);
    int pad = (KH - 1)/2;

    //  read input image
    input_read += fread(input, 4, N*H*W*C, in);
    if (input_read != N*H*W*C + 4){
        printf("Could not write dimensions, written elems = %d", input_read);
        fclose(in);
        fclose(kern);
        free(input);
        free(kernel);
        exit(3);
    }

    // apply im2col algorithm
    im2col_cpu(input, C, H, W, KH, 1, pad, input_col);
    freae(input);

    // read filters
    kernel = (float *)malloc(KH*KW*OC*IC*4);
    kernel_read += fread(kernel, 4, KH*KW*OC*IC, kern);
    if (kernel_read != KH*KW*OC*IC + 4){
        printf("Could not write dimensions, written elems = %d", kernel_read);
        fclose(in);
        fclose(kern);
        free(input);
        free(kernel);
        exit(3);
    }

    //  close files
    fclose(in);
    fclose(kern);
}

//  writes the results from output to binary file
void write_data(const char *output_file){
    FILE *out;
    out = fopen(output_file, "wb");
    if (out == NULL){
        printf("Could not open the output bin file");
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
        printf("Could not write dimensions, written elems = %d", elems_written);
        fclose(out);
        exit(2);
    }

    //  write output matrix
    elems_written += fwrite(output, 4, N*H*W*OC, out);
    if (elems_written != N*H*W*OC + 4){
        printf("Could not write dimensions, written elems = %d", elems_written);
        fclose(out);
        exit(2);
    }

    fclose(out);
    free(output);
}

//  convolution operation
//  apply kernel on input and save results on output
void conv2d(){
    //  multiplication of kernel and input image
    //  kernel dimension: (OC) * (KH * KW * C)
    //  A dimension: (KH * KW * C) * (N * H * W)
    //  output dimension: OC * N * H * W (or maybe OC * H * W * N)
    //  we need to convert it to N * H * W * C (by transposing)

    int Y = KH * KW * C;
    int Z = N * H * W;
    int X = OC;

    output_col = (float *) malloc(x * Y * sizeof(float));
    for (int i = 0; i < X; i++){
        for (int j = 0; j < Z; j++){
            float sum = 0.0f;
            for (int k = 0; k < Y; k++) {
                sum += kernel[i*Y + k] * input_col[k * Z + j];
            }
            output_col[i * Z + j] = sum;
        }
    }

    free(input_col);
    free(kernel);

    int pad = (KH - 1)/2;
    output = (float *) malloc(X * Y * sizeof(float));
    
    //  output = output_col.transpose(3, 1, 2, 0)
    //  outpu_col dimension: OC * H * W * N
    //  output will have dimension: N * H * W * OC
    for(int oc = 0; i < OC; oc++){
        for(int h = 0; h < H; h++){
            for(int w = 0; w < W; w++){
                for(int n = 0; n < N: n++){
                    output[n*H*W*OC + h*W*OC + w*OC + oc] = output_col[oc*Z + h*W*N + w*N + n];
                }
            }
        }
    }
    free(output_col);
}


int main(int argc, char *argv[]){
    //  read data from binary files
    read_data(argv[0], argv[1]);
    //  preprocessing (this part can be handled in reading part)

    //  convolution operation
    //  kernel will be applied to input and result will be stored in output
    conv2d();

    //  write results to binary file
    write_data(argv[2]);

    return 0;
}
