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
#define min(X,Y) ((X) < (Y) ? (X) : (Y))
#define max(X,Y) ((X) > (Y) ? (X) : (Y))

#include "../preprocessing/im2col.h"
// #include "matmul.h"

float *input, *kernel, *output, *output_q;
float *input_col;
int N, H, W, C;
int KH, KW, OC, IC;
float y_max = 0.0f, y_min = 0.0f;
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
    // int col = *(data + 1);
    float p[BLOCK];
    __m256 col_8, row_8, res_8;
    for (j = 0; j < BLOCK; j++){
        sum = 0.0;
        for (i = 0; i < MK; i = i + 8){
            col_8 = _mm256_loadu_ps(&kernel[row * MK + i]);
            row_8 = _mm256_loadu_ps(&input_col[j * MK + i]);
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
    printf("BLOCK is %d\n", BLOCK);
    printf("NUMTHR is %d\n", NUM_THR);

    int i, j, t;
  
    printf("We are here\n");
    pthread_t *threads;
    threads = (pthread_t*)malloc(NUM_THR*sizeof(pthread_t));
    printf("%d %d %d\n", MM, MN, MK);
    int counter = 0;
    for (i = 0; i < MM; i++){
            int *data;
            data = calloc(1,sizeof(int));
            *data = i;
            // *(data + 1) = j;
            if(pthread_create(&threads[counter++], NULL, mult, (void*)data) != 0){
                printf("Create failed at %d\n", i);
                exit(-1);
                return;
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
            }
            out[row * MN + col] = sum;
            if(y_max < sum) y_max = sum;
            if(y_min > sum) y_min = sum;
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
    clock_t start, end, ovr_start, ovr_end, q_start, q_end;
    double cpu_time_used, time_overhead, q_cpu_time_used;
    time_overhead = 0.0;

    //  Convolution operation
    start = clock();
    matmul2(output_col, X, Y, Z);
    end = clock();

    if(P == 16){
        float min_in = -25.0f;
        float max_in = 25.0f;

        int16_t max_int = (1 << 15) - 1;
        int16_t min_int = 1 << 15;

        float scale1 = ((float)(max_int - min_int)) / (max_in - min_in);
        printf("SCALE1 = %f\n", scale1);
        float scale2 = ((float)(max_int - min_int));
        printf("SCALE2 = %f\n", scale2);
        
        int16_t *q_input = (int16_t *)malloc(Y*Z*sizeof(int16_t));
        int16_t *q_kernel = (int16_t *)malloc(Y*X*sizeof(int16_t));
        int32_t *q_output = (int32_t *)malloc(Z*X*sizeof(int32_t));
        float *r_output = (float *)malloc(Z*X*sizeof(float));

        ovr_start = clock();
        for(int i = 0; i < Y*Z; i++) q_input[i] = (int16_t) round(min(max_int, max(min_int, input_col[i] * scale1)));
        for(int i = 0; i < Y*X; i++) q_kernel[i] = (int16_t) round(min(max_int, max(min_int, kernel[i] * scale2)));
        ovr_end = clock();
        time_overhead += ((double) (ovr_end - ovr_start)) / CLOCKS_PER_SEC;

        q_start = clock();
        for (int i = 0; i < X; i++){
            for (int j = 0; j < Z; j++){
                int32_t sum = 0;
                __m256i col_8, row_8, res_8;
                for (int k = 0; k < Y; k = k + 16){
                    col_8 = _mm256_loadu_si256((__m256i *)&q_kernel[i * MK + k]);
                    row_8 = _mm256_loadu_si256((__m256i *)&q_input[j * MK + k]);
                    res_8 = _mm256_mullo_epi16(col_8, row_8);
                    sum += (int16_t) _mm256_extract_epi16(res_8, 0) + (int16_t) _mm256_extract_epi16(res_8, 1)
                        +  (int16_t) _mm256_extract_epi16(res_8, 2) + (int16_t) _mm256_extract_epi16(res_8, 3)
                        +  (int16_t) _mm256_extract_epi16(res_8, 4) + (int16_t) _mm256_extract_epi16(res_8, 5)
                        +  (int16_t) _mm256_extract_epi16(res_8, 6) + (int16_t) _mm256_extract_epi16(res_8, 7)
                        +  (int16_t) _mm256_extract_epi16(res_8, 8) + (int16_t) _mm256_extract_epi16(res_8, 9)
                        +  (int16_t) _mm256_extract_epi16(res_8, 10) + (int16_t) _mm256_extract_epi16(res_8, 11)
                        +  (int16_t) _mm256_extract_epi16(res_8, 12) + (int16_t) _mm256_extract_epi16(res_8, 13)
                        +  (int16_t) _mm256_extract_epi16(res_8, 14) + (int16_t) _mm256_extract_epi16(res_8, 15);
                }
                q_output[i * Z + j] = sum;
            }
        }
        q_end = clock();
        q_cpu_time_used += ((double) (q_end - q_start)) / CLOCKS_PER_SEC;
        printf("Quantized Convolution operation took %f seconds to execute\n", q_cpu_time_used);

        ovr_start = clock();
        for(int i = 0; i < Z*X; i++) r_output[i] = (float) (q_output[i] / (scale1 * scale2));
        ovr_end = clock();
        time_overhead += ((double) (ovr_end - ovr_start)) / CLOCKS_PER_SEC;
        printf("Overhead time is: %f\n", time_overhead);

        float diff = y_max - y_min;
        float result = 0.0f;
        for(int i = 0; i < Z*X; i++){
            result +=  pow(output_col[i] - r_output[i], 2);
        }

        result = sqrt(result / Z*X) / diff;
        printf("NRMSE = %f\n", result);
        free(q_input);
        free(q_output);
        free(q_kernel);
        free(r_output);
    }

    if(P == 32){
        //round(min(max_range, max(min_range, input)) * scale_factor);
        float min_in = -25.0f;
        float max_in = 25.0f;

        int32_t max_int = 2147483647;//(1 << 31) - 1;
        int32_t min_int = -2147483647;//1 << 31;

        float scale1 = ((long) max_int - (long) min_int) / (max_in - min_in);
        printf("SCALE1 = %f\n", scale1);
        float scale2 = ((long) max_int - (long) min_int);
        printf("SCALE2 = %f\n", scale2);

        int32_t *q_input = (int32_t *)malloc(Y*Z*sizeof(int32_t));
        int32_t *q_kernel = (int32_t *)malloc(Y*X*sizeof(int32_t));
        int64_t *q_output = (int64_t *)malloc(Z*X*sizeof(int64_t));
        float *r_output = (float *)malloc(Z*X*sizeof(float));

        ovr_start = clock();
        for(int i = 0; i < Y*Z; i++) q_input[i] = (int32_t) round(min(max_int, max(min_int, input_col[i] * scale1)));
        for(int i = 0; i < Y*X; i++) q_kernel[i] = (int32_t) round(min(max_int, max(min_int, kernel[i] * scale2)));
        ovr_end = clock();
        time_overhead += ((double) (ovr_end - ovr_start)) / CLOCKS_PER_SEC;


        q_start = clock();
        for (int i = 0; i < X; i++){
            for (int j = 0; j < Z; j++){
                int64_t sum = 0;
                __m256i col_8, row_8, res_8;
                for (int k = 0; k < Y; k = k + 8){
                    col_8 = _mm256_loadu_si256((__m256i *)&q_kernel[i * MK + k]);
                    row_8 = _mm256_loadu_si256((__m256i *)&q_input[j * MK + k]);
                    res_8 = _mm256_mullo_epi32(col_8, row_8);
                    sum += (int32_t) _mm256_extract_epi32(res_8, 0) + (int32_t) _mm256_extract_epi32(res_8, 1)
                        +  (int32_t) _mm256_extract_epi32(res_8, 2) + (int32_t) _mm256_extract_epi32(res_8, 3)
                        +  (int32_t) _mm256_extract_epi32(res_8, 4) + (int32_t) _mm256_extract_epi32(res_8, 5)
                        +  (int32_t) _mm256_extract_epi32(res_8, 6) + (int32_t) _mm256_extract_epi32(res_8, 7);
                }
                q_output[i * Z + j] = sum;
            }
        }
        q_end = clock();
        q_cpu_time_used += ((double) (q_end - q_start)) / CLOCKS_PER_SEC;
        printf("Quantized Convolution operation took %f seconds to execute\n", q_cpu_time_used);


        ovr_start = clock();
        for(int i = 0; i < Z*X; i++) r_output[i] = (float) (q_output[i] / (scale1 * scale2));//pow(scale, 2));
        ovr_end = clock();
        time_overhead += ((double) (ovr_end - ovr_start)) / CLOCKS_PER_SEC;
        printf("Overhead time is: %f\n", time_overhead);

        float diff = y_max - y_min;
        float result = 0.0f;
        for(int i = 0; i < Z*X; i++){
            result +=  pow(output_col[i] - r_output[i], 2);
        }

        result = sqrt(result / Z*X) / diff;
        printf("NRMSE = %f\n", result);
        free(q_input);
        free(q_output);
        free(q_kernel);
        free(r_output);
    }


    free(input_col);
    free(kernel);

    int pad = (KH - 1)/2;
    output = (float *) malloc(X * Y * sizeof(float));

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

    double elapsed_time = conv2d(P);

    printf("Convolution operation took %f seconds to execute\n", elapsed_time); 

    // write_data("output_tensor.bin");
    free(output);
    return 0;
}
