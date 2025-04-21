#include <cuda_runtime.h>
#include <stdio.h>
#include "timer.h"

__global__ void rgb2gray_kernel(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, unsigned int width, unsigned int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        gray[idx] = (unsigned char)(0.299f * red[idx] + 0.587f * green[idx] + 0.114f * blue[idx]);
    }
}

void rgb2gray_gpu(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, unsigned int width, unsigned int height) {
    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    unsigned char *red_d, *green_d, *blue_d, *gray_d;
    cudaMalloc((void**)&red_d, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&green_d, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&blue_d, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&gray_d, width * height * sizeof(unsigned char));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);
    cudaMemcpy(red_d, red, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d, blue, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    startTime(&timer);
    rgb2gray_kernel<<<gridSize, blockSize>>>(red_d, green_d, blue_d, gray_d, width, height);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel execution time");

    // Copy result back to host
    startTime(&timer);
    cudaMemcpy(gray, gray_d, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy back to host time");

    // Free memory
    cudaFree(red_d);
    cudaFree(green_d);
    cudaFree(blue_d);
    cudaFree(gray_d);
}
