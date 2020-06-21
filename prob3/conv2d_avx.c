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
#include <float.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include <pthread.h>


#include "../preprocessing/im2col.h"
// #include "matmul.h"

float *input, *kernel, *output, *output_q;
float *input_col;
int N, H, W, C;
int KH, KW, OC, IC;
float y_max, y_min;
int BLOCK;
int NUM_THR;
int MM, MN, MK;

/*
    Convolution
*/
void *mult(void* arg)
{
    int *data = (int *)arg;
    float sum = 0.0f;
    int i = 0, j =0;

    int row = *data;
    int col = *(data + 1);
    float p[BLOCK];
    __m256 col_8, row_8, res_8;
    for (j = 0; j < BLOCK; j++){
        sum = 0.0;
        for (i = 0; i < MK; i = i + 8){
            col_8 = _mm256_loadu_ps(&kernel[row * MK + i]);
            row_8 = _mm256_loadu_ps(&input_col[(col + j) * MK + i]);
            res_8 = _mm256_mul_ps(col_8, row_8);
            for (int i = 0; i < 8; i++) sum += *(float *)&res_8[i];
        }
        p[j] = sum;
    }
    free(arg);
    pthread_exit(p);
}

void matmul(float *out, int X, int Y, int Z){
    MM = X; MN = Z; MK = Y; 
    int size = MM * MN;
    BLOCK = MN;

    NUM_THR = size / BLOCK;

    int i, j;
  
    printf("We are here\n");
    pthread_t *threads;
    threads = (pthread_t*)malloc(NUM_THR*sizeof(pthread_t));
    printf("%d %d %d\n", MM, MN, MK);
    int counter = 0;
    for (i = 0; i < MM; i++){
        for (j = 0; j < MN; j = j + BLOCK){
            int *data;
            data = calloc(2,sizeof(int));
            *data = i;
            *(data + 1) = j;
            if(pthread_create(&threads[counter++], NULL, mult, (void*)data) != 0){
                printf("Create failed at %d %d\n", i, j);
                exit(-1);
                return;
            }
        }
    }
    printf("We are here\n");
    float *res;
    for (i = 0; i < NUM_THR; i++) {
        void *k;
        pthread_join(threads[i], &k);
        res = k;
        j = 0;
        while(j < BLOCK){
            out[i * BLOCK + j] = res[j];
            j++;
        }
    }
}

void matmul2(float *out, int X, int Y, int Z){
    MM = X; MN = Z; MK = Y; 
    int size = MM * MN;
    BLOCK = MN;

    NUM_THR = size / BLOCK;

    int i, j, k, row, col;
    float sum;
  
    printf("We are here\n");
    // pthread_t *threads;
    // threads = (pthread_t*)malloc(NUM_THR*sizeof(pthread_t));
    printf("%d %d %d\n", MM, MN, MK);
    int counter = 0;
    for (row = 0; row < MM; row++){
        for (col = 0; col < MN; col++){
            __m256 col_8, row_8, res_8;
            sum = 0.0;
            for (i = 0; i < MK; i = i + 8){
                col_8 = _mm256_loadu_ps(&kernel[row * MK + i]);
                row_8 = _mm256_loadu_ps(&input_col[col * MK + i]);
                res_8 = _mm256_mul_ps(col_8, row_8);
                for (int i = 0; i < 8; i++) sum += *(float *)&res_8[i];
                out[row * MN + col] = sum;
            }
        }
    }
}

#define QUANTIZE(max_range, min_range, input, scale_factor) roundf(min(max_range, max(min_range, input)) * scale_factor)

//  normalized rootmean-square error (NRMSE)
float nrmse(){
    float diff = y_max - y_min;
    float result = 0.0f;
    for(int i = 0; i < N*H*W*OC; i++){
        result +=  pow(output[i] - output_q[i], 2);
    }

    result = sqrt(result / N*H*W*OC) / diff;
    return result;
}

//  calculate scaling factor
void scaling(int P, float *array, int size){
    // min_range = FLT_MAX;
    // max_range = FLT_MIN;
    // for(int i = 0; i < size; i++){
    //     if(array[i] < min_range) min_range = array[i];
    //     if(array[i] > max_range) max_range = array[i];
    // }

    long min_T = 1 << (P-1);
    long max_T = (1 << (P-1)) - 1;

    // const float scale_factor_from_min_side =
    //     (min_T * min_range > 0) ? min_T / min_range : max_float;
    // const float scale_factor_from_max_side =
    //     (max_T * max_range > 0) ? max_T / max_range : max_float;

    // scale_factor = min(scale_factor_from_min_side, scale_factor_from_max_side);
}


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

    //  allocate arrays
    float *input_pre = (float *)malloc(N*H*W*C*4);

    //  A is (KH*KW*C) * (N*H*W) matrix
    float *input_col_T = (float *)malloc((KH*KW*C) * (N*H*W) * 4);
    int pad = (KH - 1)/2;

    //  read input image
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
    im2col_cpu(input, C, H, W, KH, 1, pad, input_col_T);
    free(input);

    input_col = (float *)malloc((KH*KW*C) * (N*H*W) * 4);
    int cols = KW*KH*C;
    int rows = H*W*N;
    for(int x = 0; x < cols; x++){
        for(int y = 0; y < rows; y++){
            input_col[y*cols + x] = input_col_T[x*rows + y];
        }
    }
    free(input_col_T);

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
double conv2d(int P){
    int Y = KH * KW * C;
    int Z = N * H * W;
    int X = OC;

    float *output_col = (float *) malloc(X * Y * sizeof(float));
    clock_t start, end;
    double cpu_time_used;
    
    //  Convolution operation
    printf("CONV starts\n");
    start = clock();
    matmul2(output_col, X, Y, Z);
    end = clock();
    printf("CONV done\n");
    printf("%d %d %d\n", X, Y, Z);

    free(input_col);
    free(kernel);

    int pad = (KH - 1)/2;
    printf("WOW\n");
    output = (float *) malloc(X * Y * sizeof(float));
    printf("NICE\n");
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
    printf("NICE2\n");
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    return cpu_time_used;
}


int main(int argc, char *argv[]){
    const char* precision = argv[3];
    int P = 0;

    if(!strcmp(precision, "FP32")) P = 0;
    else if(!strcmp(precision, "INT32")) P = 32;
    else if(!strcmp(precision, "INT16")) P = 16;
    else{
        printf("Wrong value is given: %s, precision should be 8, 16, or 32", precision);
        exit(80);
    }

    read_data(argv[1], argv[2]);
    printf("READ data\n");

    double elapsed_time = conv2d(P);
    printf("CONV\n");

    printf("Convolution operation took %f seconds to execute\n", elapsed_time); 

    write_data("output_tensor.bin");
    // free(output);
    return 0;
}
