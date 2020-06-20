#include <iostream>
#include <algorithm>
#include <stdio.h>
using namespace std;

#define BLOCK_SIZE 16
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString( err ), file, line);
        exit(EXIT_FAILURE);
    }
}


/*
    Convolution
*/
__global__ void gpu_multABtoC(float *a,float *b, float *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

extern "C"{
    void matmul(float *C, float *A, float *B, int m, int n, int k)
    {
        // Allocate memory space on the device 
        float *dev_a, *dev_b, *dev_c;
        cudaMalloc((void **) &dev_a, sizeof(float)*m*n);
        cudaMalloc((void **) &dev_b, sizeof(float)*n*k);
        cudaMalloc((void **) &dev_c, sizeof(float)*m*k);

        // copy matrix A and B from host to device memory
        cudaMemcpy(dev_a, A, sizeof(float)*m*n, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, B, sizeof(float)*n*k, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_c, C, sizeof(float)*m*k, cudaMemcpyHostToDevice);

        unsigned int gridev_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int gridev_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimGrid(gridev_cols, gridev_rows);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
        // Launch kernel 
        gpu_multABtoC<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, m, n, k);    

        // Transefr results from device to host 
        cudaMemcpy(C, dev_c, sizeof(float)*m*k, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // free memory
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
    }
}