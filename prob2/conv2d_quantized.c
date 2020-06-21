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

#include "../preprocessing/im2col.h"


#define min(X,Y) ((X) < (Y) ? (X) : (Y))
#define max(X,Y) ((X) > (Y) ? (X) : (Y))


float *input, *kernel, *output, *output_q;
float *input_col;
int N, H, W, C;
int KH, KW, OC, IC;
float y_max, y_min;
// float scale_factor, min_range, max_range;
// float max_float = FLT_MAX;

// int32_t *input32, *kernel32, *output32;
// int16_t *input16, *kernel16, *output16;
// int8_t *input8, *kernel8, *output8;

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
    input_col = (float *)malloc((KH*KW*C) * (N*H*W) * 4);
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
    float y_max = FLT_MIN;
    float y_min = FLT_MAX;
    
    start = clock();
    for (int i = 0; i < X; i++){
        for (int j = 0; j < Z; j++){
            float sum = 0.0f;
            for (int k = 0; k < Y; k++) {
                sum += kernel[i*Y + k] * input_col[k * Z + j];
            }
            output_col[i * Z + j] = sum;
            if(y_max < sum) y_max = sum;
            if(y_min > sum) y_min = sum;
        }
    }
    end = clock();
    printf("MAX and MIN results \ny_max = %f, \ny_min = %f\n", y_max, y_min);

    //  Quantization starts
    //  start from 0 because of padding
    if(P == 8){
        float min_in = -30.0f;
        float max_in = 30.0f;

        int8_t max_int = 127;
        int8_t min_int = -128;

        float scale1 = round(min((float)max_int / max_in, (float)min_int / min_in));
        printf("SCALE = %f\n", scale1);
        // float scale2 = max_int - min_int;
        float scale2 = 64;
        
        int8_t *q_input = (int8_t *)malloc(Y*Z*sizeof(int8_t));
        int8_t *q_kernel = (int8_t *)malloc(Y*X*sizeof(int8_t));
        int8_t *q_output = (int8_t *)malloc(Z*X*sizeof(int8_t));
        float *r_output = (float *)malloc(Z*X*sizeof(float));
        ovr_start = clock();
        for(int i = 0; i < Y*Z; i++) q_input[i] = (int8_t) round(min(max_int, max(min_int, input_col[i] * scale1)));
        for(int i = 0; i < Y*X; i++) q_kernel[i] = (int8_t) round(min(max_int, max(min_int, kernel[i] * scale2)));
        ovr_end = clock();
        time_overhead += ((double) (ovr_end - ovr_start)) / CLOCKS_PER_SEC;


        q_start = clock();
        for (int i = 0; i < X; i++){
            for (int j = 0; j < Z; j++){
                int8_t sum = 0;
                for (int k = 0; k < Y; k++) {
                    sum += q_kernel[i*Y + k] * q_input[k * Z + j];
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
    
    if(P == 16){
        float min_in = -25.0f;
        float max_in = 25.0f;

        int16_t max_int = (1 << 15) - 1;
        int16_t min_int = 1 << 15;

        float scale1 = ((float)(max_int - min_int)) / (max_in - min_in);
        printf("SCALE = %f\n", scale1);
        float scale2 = ((float)(max_int - min_int));
        
        int16_t *q_input = (int16_t *)malloc(Y*Z*sizeof(int16_t));
        int16_t *q_kernel = (int16_t *)malloc(Y*X*sizeof(int16_t));
        int16_t *q_output = (int16_t *)malloc(Z*X*sizeof(int16_t));
        float *r_output = (float *)malloc(Z*X*sizeof(float));

        q_start = clock();
        for(int i = 0; i < Y*Z; i++) q_input[i] = (int16_t) (input_col[i] * scale1);
        for(int i = 0; i < Y*X; i++) q_kernel[i] = (int16_t) (kernel[i] * scale2);
        ovr_end = clock();
        time_overhead += ((double) (ovr_end - ovr_start)) / CLOCKS_PER_SEC;

        q_start = clock();
        for (int i = 0; i < X; i++){
            for (int j = 0; j < Z; j++){
                int16_t sum = 0;
                for (int k = 0; k < Y; k++) {
                    sum += q_kernel[i*Y + k] * q_input[k * Z + j];
                }
                q_output[i * Z + j] = sum;
            }
        }
        q_end = clock();
        q_cpu_time_used += ((double) (q_end - q_start)) / CLOCKS_PER_SEC;
        printf("Quantized Convolution operation took %f seconds to execute\n", q_cpu_time_used);

        q_start = clock();
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

    if(P == 32){
        float min_in = -25.0f;
        float max_in = 25.0f;

        long max_int = 0x7FFFFFFF;//(1 << 31) - 1;
        long min_int = 0x80000000;//1 << 31;

        float scale1 = ((float) max_int) / (max_in - min_in);
        printf("SCALE = %f\n", scale1);
        float scale2 = ((double)(max_int - min_int));

        int32_t *q_input = (int32_t *)malloc(Y*Z*sizeof(int32_t));
        int32_t *q_kernel = (int32_t *)malloc(Y*X*sizeof(int32_t));
        int32_t *q_output = (int32_t *)malloc(Z*X*sizeof(int32_t));
        float *r_output = (float *)malloc(Z*X*sizeof(float));

        q_start = clock();
        for(int i = 0; i < Y*Z; i++) q_input[i] = (int32_t) (input_col[i] * scale1);
        for(int i = 0; i < Y*X; i++) q_kernel[i] = (int32_t) (kernel[i] * scale2);
        ovr_end = clock();
        time_overhead += ((double) (ovr_end - ovr_start)) / CLOCKS_PER_SEC;

        q_start = clock();
        for (int i = 0; i < X; i++){
            for (int j = 0; j < Z; j++){
                int32_t sum = 0;
                for (int k = 0; k < Y; k++) {
                    sum += q_kernel[i*Y + k] * q_input[k * Z + j];
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
    if(!strcmp(precision, "32")) P = 32;
    else if(!strcmp(precision, "16")) P = 16;
    else if(!strcmp(precision, "8")) P = 8;
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
