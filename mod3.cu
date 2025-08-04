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
#include <stdint.h>

/* Array size for the sort (power of two) */
/* Final benchmarking uses 2^30 values */
#define ARRAY_SIZE (1<<30)
/* Threads per block */
#define THREADS 256
/* blocks depend on how many pairs each thread handles */
#define PAIRS_PER_THREAD 4
#define BLOCKS ((ARRAY_SIZE/(2*PAIRS_PER_THREAD) + THREADS - 1) / THREADS)
#define NUM_VALS ARRAY_SIZE

static inline uint32_t random_uint()
{
  return ((uint32_t)rand() << 16) ^ (uint32_t)rand();
}

void array_print(uint32_t *arr, int length)
{
  for (int i = 0; i < length; ++i) {
    printf("%u ",  arr[i]);
  }
  printf("\n");
}

void array_fill(uint32_t *arr, int length)
{
  srand(time(NULL));
  for (int i = 0; i < length; ++i) {
    arr[i] = random_uint();
  }
}

bool verify_sorted(const uint32_t *arr, int length) {
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
 * each thread processes per j/set iteration.  The kernel below performs a
 * single bitonic step and is used when cross-block synchronization is
 * required.
 */
__global__ void bitonic_step(uint32_t *dev_values, unsigned int j, unsigned int k)
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

    uint32_t val1 = dev_values[index1];
    uint32_t val2 = dev_values[index2];

    if (ascending ? val1 > val2 : val1 < val2) {
      dev_values[index1] = val2;
      dev_values[index2] = val1;
    }
  }
}

/*
 * When all data for a given j/set value fits within a block, multiple j/set
 * iterations can be combined inside one kernel using __syncthreads().  This
 * kernel incorporates the additional per-thread loop from modification #3.
 */
__global__ void bitonic_block(uint32_t *dev_values, unsigned int j_start,
                              unsigned int j_end, unsigned int k)
{
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int totalThreads = blockDim.x * gridDim.x;

  for (unsigned int j = j_start; j >= j_end; j >>= 1) {
    unsigned int groupSize = j << 1;

    for (int p = 0; p < PAIRS_PER_THREAD; ++p) {
      unsigned int id = tid + p * totalThreads;
      if (id >= NUM_VALS / 2)
        continue;

      unsigned int index1 = (id / j) * groupSize + (id % j);
      unsigned int index2 = index1 + j;

      bool ascending = ((index1 & k) == 0);

      uint32_t val1 = dev_values[index1];
      uint32_t val2 = dev_values[index2];

      if (ascending ? val1 > val2 : val1 < val2) {
        dev_values[index1] = val2;
        dev_values[index2] = val1;
      }
    }

    __syncthreads();
  }
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(uint32_t *values, bool timing = true)
{
  uint32_t *dev_values;
  size_t size = NUM_VALS * sizeof(uint32_t);

  cudaMalloc((void**) &dev_values, size);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */

  using clock_type = std::chrono::high_resolution_clock;
  auto start = clock_type::now();

  unsigned int k;
  /* Major step */
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    unsigned int j = k >> 1;

    /* Handle j values that span multiple blocks and thus require
       separate kernel launches for global synchronization. */
    while (j > THREADS * PAIRS_PER_THREAD) {
      bitonic_step<<<blocks, threads>>>(dev_values, j, k);
      j >>= 1;
    }

    /* Remaining j values fit within a block and are combined inside one kernel */
    bitonic_block<<<blocks, threads>>>(dev_values, j, 1, k);
  }

  cudaDeviceSynchronize();
  auto t2 = clock_type::now();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  if (timing) {
    std::chrono::duration<double, std::milli> ms = t2 - start;
    printf("Kernel wall time elapsed: %g ms\n", ms.count());
  }

  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaFree(dev_values);
}

bool verify_unique(uint32_t *arr, int length) {
  for (int i = 0; i < length; ++i) {
    arr[i] = (uint32_t)(length - 1 - i);
  }
  bitonic_sort(arr, false);
  for (int i = 0; i < length; ++i) {
    if (arr[i] != (uint32_t)i) {
      printf("Value mismatch at index %d\n", i);
      return false;
    }
  }
  printf("All values present exactly once\n");
  return true;
}

int main(void)
{
  uint32_t *values = (uint32_t*) malloc(NUM_VALS * sizeof(uint32_t));
  array_fill(values, NUM_VALS);

  bitonic_sort(values, true);
  verify_sorted(values, NUM_VALS);
  verify_unique(values, NUM_VALS);

  free(values);
  return 0;
}
