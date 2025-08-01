/*
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc -arch=sm_11 bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 * License: BSD 3
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <chrono>

/* Array size must be a power of 2 */
/* Use 2^30 values for the final timing comparison. */
#define ARRAY_SIZE (1<<30)
/* Threads per block */
#define THREADS 256
/* Each thread handles a pair of values so total threads is ARRAY_SIZE/2 */
#define BLOCKS ((ARRAY_SIZE/2 + THREADS - 1) / THREADS)
#define NUM_VALS ARRAY_SIZE

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

void array_print(float *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
}

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}

bool verify_sorted(const float *arr, int length) {
  for (int i = 0; i < length - 1; ++i) {
    if (arr[i] > arr[i + 1]) {
      printf("Sort error at indexes %d and %d\n", i, i + 1);
      return false;
    }
  }
  printf("Array sorted correctly\n");
  return true;
}

/*
 * Modification #1 - prevent throwing away threads. Each CUDA thread now works
 * on exactly one pair of indexes without the ixj > i check. The new indexing
 * logic computes unique pairs for every thread.
 */
__global__ void bitonic_sort_step(float *dev_values, int j, int k)
{
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

  unsigned int groupSize = j << 1;       /* 2 * j */
  unsigned int index1   = (tid / j) * groupSize + (tid % j);
  unsigned int index2   = index1 + j;

  bool ascending = ((index1 & k) == 0);

  float val1 = dev_values[index1];
  float val2 = dev_values[index2];

  if (ascending ? val1 > val2 : val1 < val2) {
    dev_values[index1] = val2;
    dev_values[index2] = val1;
  }
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(float *values)
{
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);

  cudaMalloc((void**) &dev_values, size);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */

  using clock_type = std::chrono::high_resolution_clock;
  auto t1 = clock_type::now();

  int j, k;
  /* Major step */
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
    }
  }

  cudaDeviceSynchronize();
  auto t2 = clock_type::now();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  std::chrono::duration<double, std::milli> ms = t2 - t1;
  printf("Kernel wall time elapsed: %g ms\n", ms.count());

  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaFree(dev_values);
}

int main(void)
{
  float *values = (float*) malloc(NUM_VALS * sizeof(float));
  array_fill(values, NUM_VALS);

  bitonic_sort(values);
  verify_sorted(values, NUM_VALS);

  free(values);
  return 0;
}
