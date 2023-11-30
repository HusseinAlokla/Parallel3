% % writefile matrix_mul.cu
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

    const int M = 1000;
const int N = 2000;
const int K = 2000;

// Tile dimensions
const int TILE_WIDTH = 16;

// CUDA kernel for matrix multiplication with tiling
__global__ void matrixMulTiled(float *A, float *B, float *C, int m, int n, int k)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identify the row and column of the C element to work on
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    float Cvalue = 0.0;

    // Loop over the tiles of the input matrices
    for (int t = 0; t < (k - 1) / TILE_WIDTH + 1; ++t)
    {
        // Collaborative loading of A and B tiles into shared memory
        if (row < m && t * TILE_WIDTH + tx < k)
            As[ty][tx] = A[row * k + t * TILE_WIDTH + tx];
        else
            As[ty][tx] = 0.0;

        if (col < n && t * TILE_WIDTH + ty < k)
            Bs[ty][tx] = B[(t * TILE_WIDTH + ty) * n + col];
        else
            Bs[ty][tx] = 0.0;

        __syncthreads();

        // Compute the product of the tile
        for (int i = 0; i < TILE_WIDTH; ++i)
            Cvalue += As[ty][i] * Bs[i][tx];

        __syncthreads();
    }

    // Write the result to the output matrix
    if (row < m && col < n)
        C[row * n + col] = Cvalue;
}

int main()
{
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    for (int i = 0; i < M * K; ++i)
    {
        h_A[i] = i;
    }

    for (int i = 0; i < K * N; ++i)
    {
        h_B[i] = 2 * i;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Set grid and block dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(ceil(N / float(TILE_WIDTH)), ceil(M / float(TILE_WIDTH)), 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start);

    // Launch the tiled kernel
    matrixMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate and print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time: " << milliseconds << " ms" << std::endl;

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
