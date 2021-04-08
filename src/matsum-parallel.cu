#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define T 32

__global__ void matrix_sum(int *A, int *B, int *C, int N, int M) {
  int col = T * blockIdx.x + threadIdx.x;
  int row = T * blockIdx.y + threadIdx.y;

  if (row < N && col < M) {
    C[row * M + col] = A[row * M + col] + B[row * M + col];
  }
}

int main(int argc, char **argv) {
  int *A, *B, *C;
  int *d_A, *d_B, *d_C;
  int i, j;
  double t;

  // Input
  int rows, cols, size;
  FILE *input;

  if (argc < 2) {
    fprintf(stderr, "Error: missing path to input file\n");
    return EXIT_FAILURE;
  }

  if ((input = fopen(argv[1], "r")) == NULL) {
    fprintf(stderr, "Error: could not open file\n");
    return EXIT_FAILURE;
  }

  fscanf(input, "%d", &rows);
  fscanf(input, "%d", &cols);

  size = sizeof(int) * rows * cols;
  // Allocate memory on the host
  A = (int *)malloc(size);
  B = (int *)malloc(size);
  C = (int *)malloc(size);

  // Initialize memory
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      A[i * cols + j] = B[i * cols + j] = i + j;
    }
  }

  // Copy data to device
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

  dim3 dimGrid(ceil((float) cols / T), ceil((float) rows / T), 1);
  dim3 dimBlock(T, T, 1);

  // Compute matrix sum on device
  // Leave only the kernel and synchronize inside the timing region!
  t = omp_get_wtime();
  matrix_sum<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, rows, cols);
  cudaDeviceSynchronize();
  t = omp_get_wtime() - t;

  // Copy data back to host
  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  long long int sum = 0;

  // Keep this computation on the CPU
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      sum += C[i * cols + j];
    }
  }

  fprintf(stdout, "%lli\n", sum);
  fprintf(stderr, "%lf\n", t);

  free(A); free(B); free(C);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
