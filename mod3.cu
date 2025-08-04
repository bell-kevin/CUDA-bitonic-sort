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

/* Array size for the sort (power of two) */
/* Final benchmarking uses 2^30 values */
#define ARRAY_SIZE (1<<30)
/* Threads per block */
#define THREADS 256
/* blocks depend on how many pairs each thread handles */
#define PAIRS_PER_THREAD 4
#define BLOCKS ((ARRAY_SIZE/(2*PAIRS_PER_THREAD) + THREADS - 1) / THREADS)
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
 * Modification #3 - have each thread operate on multiple pairs of values to
 * increase cache reuse. The PAIRS_PER_THREAD constant controls how many pairs
 * each thread processes per j/set iteration.
 */
__global__ void bitonic_sort_step(float *dev_values, int j, int k)
{
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

  for (int p = 0; p < PAIRS_PER_THREAD; ++p) {
    unsigned int id = tid + p * blockDim.x * gridDim.x;

    if (id >= NUM_VALS / 2)
      return;

    unsigned int groupSize = j << 1;
    unsigned int index1   = (id / j) * groupSize + (id % j);
    unsigned int index2   = index1 + j;

    bool ascending = ((index1 & k) == 0);

    float val1 = dev_values[index1];
    float val2 = dev_values[index2];

    if (ascending ? val1 > val2 : val1 < val2) {
      dev_values[index1] = val2;
      dev_values[index2] = val1;
    }
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
  auto start = clock_type::now();
  int stages = 0;
  for (size_t tmp = NUM_VALS; tmp > 1; tmp >>= 1) ++stages;
  int total_steps = stages * (stages + 1) / 2;
  int completed_steps = 0;
  int last_percent = -1;

  int j, k;
  /* Major step */
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
      cudaDeviceSynchronize();
      completed_steps++;
      auto now = clock_type::now();
      double elapsed = std::chrono::duration<double>(now - start).count();
      int percent = completed_steps * 100 / total_steps;
      while (percent > last_percent) {
        double est_total = elapsed / completed_steps * total_steps;
        double remaining = est_total - elapsed;
        last_percent++;
        printf("Progress: %d%% done, ETA %.2f s\n", last_percent, remaining);
      }
    }
  }

  auto t2 = clock_type::now();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  std::chrono::duration<double, std::milli> ms = t2 - start;
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
