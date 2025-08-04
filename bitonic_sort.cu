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

/* Every thread gets exactly one value in the unsorted array. */
/*
 * For the final timing comparison the README requests sorting 2^30 values.
 * With 512 threads per block this requires 2^21 blocks.
 */
#define THREADS 512       // 2^9
#define BLOCKS 2097152    // 2^21
#define NUM_VALS ((size_t)THREADS * BLOCKS)

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

__global__ void bitonic_sort_step(float *dev_values, unsigned int j, unsigned int k)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
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

  unsigned int j, k;
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
