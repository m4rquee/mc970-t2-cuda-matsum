#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define T 32
#define T_exp 5 // T = 1 << T_exp

#define CUDACHECK(cmd) { \
  cudaError_t e = cmd; \
  if (e != cudaSuccess) { \
    printf("Failed: Cuda error %s:%d '%s'\n", \
      __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(EXIT_FAILURE); \
  } \
}

__global__ void matrix_sum(int *A, int *B, int *C, int N, int M) {
  int row = (blockIdx.y << T_exp) + threadIdx.y;
  int col = (blockIdx.x << T_exp) + threadIdx.x;

  asm(
  "{"
    ".reg .pred %p;"
    "setp.ge.s32 %p, %0, %2;" // set %p with row >= N 
    "@%p mov.s32 %0, 0;" // conceptually: row = (row >= N) ? 0 : row
    "setp.ge.s32 %p, %1, %3;" // set %p with col >= M
    "@%p mov.s32 %1, 0;" // conceptually: col = (col >= M) ? 0 : col
  "}"
  : "+r"(row), "+r"(col)
  : "r"(N), "r"(M));

  int i = row * M + col;
  C[i] = A[i] + B[i];
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
  CUDACHECK(cudaMalloc(&d_A, size));
  CUDACHECK(cudaMalloc(&d_B, size));
  CUDACHECK(cudaMalloc(&d_C, size));
  CUDACHECK(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

  dim3 dimGrid(ceil((float) cols / T), ceil((float) rows / T), 1);
  dim3 dimBlock(T, T, 1);

  // Compute matrix sum on device
  // Leave only the kernel and synchronize inside the timing region!
  t = omp_get_wtime();
  matrix_sum<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, rows, cols);
  CUDACHECK(cudaDeviceSynchronize());
  t = omp_get_wtime() - t;

  // Copy data back to host
  CUDACHECK(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaDeviceSynchronize());

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
  CUDACHECK(cudaFree(d_A));
  CUDACHECK(cudaFree(d_B));
  CUDACHECK(cudaFree(d_C));
}
