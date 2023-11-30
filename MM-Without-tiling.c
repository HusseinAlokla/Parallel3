% % writefile matrix.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

        // Matrix multiplication kernel
        __global__ void
        matrixMul(int *a, int *b, int *c, int m, int n, int p)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p)
    {
        int sum = 0;
        for (int i = 0; i < n; ++i)
        {
            sum += a[row * n + i] * b[i * p + col];
        }
        c[row * p + col] = sum;
    }
}

// Function to initialize matrices with random values
void initMatrix(int *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            matrix[i * cols + j] = rand() % 10; // Random values between 0 and 9
        }
    }
}

int main()
{
    // Set matrix dimensions
    int m = 1000; // Number of rows in A
    int n = 2000; // Number of columns in A and rows in B
    int p = 3000; // Number of columns in B

    // Allocate host memory
    int *h_a = (int *)malloc(m * n * sizeof(int));
    int *h_b = (int *)malloc(n * p * sizeof(int));
    int *h_c = (int *)malloc(m * p * sizeof(int));

    // Initialize matrices with random values
    initMatrix(h_a, m, n);
    initMatrix(h_b, n, p);

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, m * n * sizeof(int));
    cudaMalloc((void **)&d_b, n * p * sizeof(int));
    cudaMalloc((void **)&d_c, m * p * sizeof(int));

    // Copy matrices from host to device
    cudaMemcpy(d_a, h_a, m * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * p * sizeof(int), cudaMemcpyHostToDevice);

    // Set up the grid and block dimensions
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((p + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y, 1);

    // Record start time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch the matrix multiplication kernel
    matrixMul<<<gridDim, blockDim>>>(d_a, d_b, d_c, m, n, p);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy result matrix from device to host
    cudaMemcpy(h_c, d_c, m * p * sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate and print the execution time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Matrix multiplication time: %.2f ms\n", milliseconds);

    // Free device and host memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
