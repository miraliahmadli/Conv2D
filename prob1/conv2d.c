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
using namespace std;

float *input, *kernel, *output;
int N, H, W, C;
int KH, KW, OC, IC;
int OH, OW;

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
    kernel = (float *)malloc(KH*KW*OC*IC*4);

    //  read arrays
    input_read += fread(input, 4, N*H*W*C, in);
    if (input_read != N*H*W*C + 4){
        printf("Could not write dimensions, written elems = %d", input_read);
        fclose(in);
        fclose(kern);
        free(input);
        free(kernel);
        exit(3);
    }

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
    elems_written += fwrite(&OH, 4, 1, out);
    elems_written += fwrite(&OW, 4, 1, out);
    elems_written += fwrite(&OC, 4, 1, out);
    if (elems_written != 4){
        printf("Could not write dimensions, written elems = %d", elems_written);
        fclose(out);
        exit(2);
    }

    //  write output matrix
    elems_written += fwrite(output, 4, N*OH*OW*OC, out);
    if (elems_written != N*OH*OW*OC + 4){
        printf("Could not write dimensions, written elems = %d", elems_written);
        fclose(out);
        exit(2);
    }

    fclose(out);
}

//  convolution operation
//  apply kernel on input and save results on output
void conv2d(){

}


int main(){
    //  read data from binary files

    //  preprocessing (this part can be handled in reading part)

    //  convolution operation
    //  kernel will be applied to input and result will be stored in output


    //  write results to binary file

    //  free memory
    free(input);
    free(kernel);
    free(output);
    return 0;
}
