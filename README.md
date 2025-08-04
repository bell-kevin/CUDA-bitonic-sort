<a name="readme-top"></a>

# 
GPU Bitonic Sort.

Overall goal: Implement optimized CUDA code by improving a naïve algorithm.

Secondary goals:

● Gain experience in more complicated indexing problems.

● Know how to correctly group parallel threads with barriers to prevent race conditions

Background:

Academia sometimes creates an image of a professor and students high on an ivory tower, effortlessly turning theory
into pseudocode and into C++ code. In reality, we often just use Google or AI, find an existing algorithm, then clean
that up. This assignment does not deny this reality.

A fairly straightforward description of a bitonic sort can be given in both text and in pictures.

The single threaded CPU
implementation is also fairly clean and straightforward. 

Consider the following figure:

This approach can represent 8 CPU threads operating on 16 elements. For example, the first two black lines are
thread 0, the second two black lines are thread 1, and so on. Thread 0 compares arr[1] < arr[0], and if true, swaps
them. Thread 1 compares arr[2] < arr[3], and if so, swaps them. This continues for 8 total < comparisons. Then the
next phase starts. Thread 0 compares arr[2] < arr[0], and if so, swaps them. Thread 1 compares arr[3] < arr[1] and
swaps if needed. This continues for 8 total < comparisons. The figure implies that a barrier is needed, the third set of 8
< comparisons cannot start until the all prior < comparisons complete. Overall, this figure has 10 phases of 8 <
comparisons, or 80 < total. The figure also requires 10 barriers.
However, using 8 CPU threads is a poor implementation. A better implementation would be 1 CPU thread running
SIMD operations.
Since GPUs are SIMD by default, this can be implemented on GPUs too. The GPU implementation is not so
straightforward, especially when efficiency and avoiding waste are desired.

An existing implementation of a CUDA bitonic sort is given here: https://gist.github.com/mre/1392067

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

/* Every thread gets exactly one value in the unsorted array. */
#define THREADS 512 // 2^9
#define BLOCKS 32768 // 2^15
#define NUM_VALS THREADS*BLOCKS

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

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

__global__ void bitonic_sort_step(float *dev_values, int j, int k)
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

  int j, k;
  /* Major step */
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
    }
  }
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaFree(dev_values);
}

int main(void)
{
  clock_t start, stop;

  float *values = (float*) malloc( NUM_VALS * sizeof(float));
  array_fill(values, NUM_VALS);

  start = clock();
  bitonic_sort(values); /* Inplace */
  stop = clock();

  print_elapsed(start, stop);
}


This algorithm has a few problems, namely:

1) The first CUDA kernel’s if statement throws away half of allocated GPU threads to keep its indexing logic
simple. This is wasteful and affects resources and performance.
2) The algorithm relaunches the CUDA kernel each step of the bitonic sort, but this relaunching is excessive.
3) The algorithm reads and writes to global memory once in each CUDA kernel, lowering cache reuse.
4) Each thread is responsible for swapping a single pair of values and nothing more. A thread has no additional
function beyond that.

Instructions:

This assignment requires that you incrementally improve the algorithm in three ways and report the results.

Start by verifying you can download, compile, and run the application.

The following should compile it on the server:

/usr/local/cuda-12/bin/nvcc -arch=sm_86 bitonic_sort.cu -o bitonic_sort.x

Modify the code so that it times the kernel execution only, and not the copy in/copy out times.

// copy data into GPU

auto t1 = std::chrono::high_resolution_clock::now();

// code to time

cudaDeviceSynchronize()

auto t2 = std::chrono::high_resolution_clock::now();

// check for error

cudaError_t error = cudaGetLastError();

if (error != cudaSuccess) {

// print the CUDA error message and exit

printf("CUDA error: %s\n", cudaGetErrorString(error));

exit(-1);

}

std::chrono::duration<double, std::milli> fp_ms = t2 - t1;

printf("Kernel wall time elapsed: %g ms\n", fp_ms.count());

// copy data out of GPU

Choose an array size that has a power of 2 (this bitonic sort can’t work in other sizes). The base version and three
modifications should be saved in standalone files and submitted.

Start by implementing a check to verify it sorts all values correctly. This should occur after the sort is done and data is
copied back to host memory. If when any value at index i is greater than a value at index i + 1, print that an issue was
found and what indexes they were. This will help greatly for later debugging. If no issues were found, print that it
sorted correctly.

Modification #1 – Preventing throwing away threads:

This step can feel like indexing hell. Thus, proceed carefully by cleaning up the code, understanding the code in detail,
and documenting each step. The first part of this Modification #1 is prepping your code and documenting it. Create
your own algorithm design spreadsheet whose values mirror what the kernel is computing for its values. Some steps I
found helpful:

• Rework the logic to enable writing the array size as one variable value, then computing the blocks off that. This
allows you to more easily change the array size during testing.

• Rename the variables so that they are a bit more intuitive (such as round instead of k, set instead of j,
threadID instead of i, etc.)

• Use printf inside the kernel to indicate the round, set, array indexes, and if it’s doing an ascending/min or
descending/max.

• Add a cudaDeviceSynchronize() after every kernel call so that everything in the GPU printf buffer is
returned and prints after each kernel invocation (you will want to comment this out for your timing later).

• Set the starter program to something small, such as a small array requiring only 16 CUDA threads and 1 block.
Run the program and obtain a list of values. Populate your spreadsheet with these values.

• Save this program as a temporary .cu file (such as sort-printf.cu). You will likely come back to it to compare its
printf statements with the modified code’s printf you write later.

• In your algorithm design spreadsheet, visually mark which CUDA threads are kept and which ones are thrown
away (such as highlighting kept rows).

At this point, your spreadsheet should have columns for the threadID, round, set, and what indexes that thread
compared (two indexes or zero indexes).
The statement if ((ixj)>i) { doesn’t use half of the CUDA threads. Rework your logic so that every CUDA thread
participates. Your goal is to get each CUDA thread to work on two indexes total. That will require half as many CUDA
threads to work on the same array size (or said another way, you can keep the number of threads per CUDA block the
same, and just have your kernel now work with twice as many index values as before).

• Using the spreadsheet, create a second set of columns. In these columns, find a way to generate all array
index compare and swaps so that every thread now participates on two indexes, and does the comparison
properly. I personally changed the i variable to threadID, changed [i] to [index1], and [ixj] to [index2].
I was able to compute index1, index2, and the ascending/descending from the k/round, j/set, and
i/threadID variables only by using addition, multiplication, and integer division (which is
QUOTIENT(var1, var2) in spreadsheet formulas). Bitwise instructions will probably be even more efficient.
Note that the rows won’t match with values. Previously half of the threads did no work. But in this new set of
columns, all threads work on two indexes. Just make sure that all indexes are properly covered in the new
indexes.

• Once the new spreadsheet columns demonstrate proper logic, translate this into CUDA/C++ code.

Update
accordingly the printf statements for debugging/comparison.

• Update the kernel invocation to account for the fact that somehow you need only half as many threads as
before (or said another way, each kernel can work with twice as many index values).

• Test your code and ensure it sorts. Carefully dial up the array size by increasing powers of two. 

Test and
ensure it works. Increase the array size so that two CUDA blocks are required. Test and ensure it works.

Continue to increase the array size until it matches your initial comparison benchmark.
Obtain timings. Make sure you remove all printf statements and all cudaDeviceSynchronize() prior to obtaining
this timing (except the last one before the timer records t2). Save this .cu as a separate finished file (mod1.cu) as it’s
needed for your submission.

Modification #2 – Merge together kernels

This step requires you split apart the j/set loop and place some iterations in CPU code and some iterations in GPU
code. The overall concept is that when one kernel’s block of threads work with indexes that are independent of
another kernel’s block of threads, then the j/set loop can now iterate inside the kernel, rather than requiring multiple
kernel calls.

The below figure is the goal. Here the array is 32 elements, each CUDA thread block is 8 threads, and each thread
works on one element of the array (all threads do work). Each blue box is a kernel call. Each purple box is a block.
Each green line is a __syncthreads() barrier to prevent a race condition. Each kernel gets two thread blocks.

(Unused optimization: Technically the first 4 kernels (blue boxes) could be rearranged into a different pattern of two
kernels as there aren’t any dependencies. In other words, in the figure’s first 4 kernels, data between [0,15] never
interacts with data between [16,31].)
The visual perspective can be generalized. If a CUDA thread block is 16 threads, then the number of blocks per kernel
is 1, then the final two kernels (blue boxes) can be merged into one, and one block (purple box) is used per kernel,
and the final kernel would have 4 __syncthreads calls (green lines).

Another way to see the pattern is observing j loop iterations in a GPU kernel. Here each thread block is 8:

1 iteration (j = 1)

2 iterations (j = 2, 1)

3 iterations (j = 4, 2, 1)

4 iterations (j = 8, 4, 2, 1)

1 iteration (j = 16)

4 iterations (j = 8,4,2,1) < - - - - - - The figure ends here

1 iteration (j = 32)

1 iteration (j = 16)

4 iterations (j = 8,4,2,1)

1 iteration (j = 64)

1 iteration (j = 32)

1 iteration (j = 16)

4 iterations (j = 8,4,2,1)

... and so on ...

For 512 threads per block it becomes:

1 iteration (j = 1)

2 iterations (j = 2, 1)

3 iterations (j = 4, 2, 1)

4 iterations (j = 8, 4, 2, 1)

5 iterations (j = 16, 8, 4, 2, 1)

6 iterations (j = 32, 16, 8, 4, 2, 1)

7 iterations (j = 64, 32, 16, 8, 4, 2, 1)

8 iterations (j = 128, 64, 32, 16, 8, 4, 2, 1)

9 iterations (j = 256, 128, 64, 32, 16, 8, 4, 2, 1)

1 iteration (j = 512)

9 iterations (j = 256, 128, 64, 32, 16, 8, 4, 2, 1)

1 iteration (j = 1024)

1 iteration (j = 512)

9 iterations (j = 256, 128, 64, 32, 16, 8, 4, 2, 1)

1 iteration (j = 2048)

1 iteration (j = 1024)

1 iteration (j = 512)

9 iterations (j = 256, 128, 64, 32, 16, 8, 4, 2, 1)

... and so on ...

This step again can feel like indexing hell. I recommend the following:

• Start by sizing down the array so that only one thread block is used. Ensure that each kernel calls only 1 block.

For example, an array of 32 indexes and thread blocks is 16. Get this to work before attempting larger arrays.

• Implement a j/set loop inside of the GPU code. At the last line inside the GPU’s j loop, add __syncthreads()
to prevent a race condition where one warp within a block may get ahead and compute before other warps
have started.

• Compute the GPU j/set loop start value and j/set loop end value in CPU code. Pass those as arguments into
the kernel and modify the loop to use those parameters. For example, when k = 16, then j’s GPU start and end
indexes are 8 and 1, so that it can iterate 8, 4, 2, and 1. But when k = 32, then j’s GPU start and end indexes
should be 16 and 16.

• I found this last step particularly tricky even though the resulting solution ended up simple and clean. What
helped me was to rewrite the CPU’s j for loop into a do while loop. The do while loop ensured the kernel is at
least called once. I also modified the loop to have a new termination condition.

• Add printf statements after the GPU loop to indicate k/round, the current j/set iteration value, and other
values you deem helpful. Add this printf statement inside an if statement that only executes if threadId.x or
threadID is 0. You don’t want this printf displaying for every CUDA thread, that’s unnecessary info. Add back
in the cudaDeviceSynchronize() after the kernel call so your printf statements are copied back to host and
displayed after each kernel invocation.

• Compile, run, and verify the sort is still correct.

• Increase the array size until it maxes out one CUDA block. Verify it sorts correctly.

• Change values so that two blocks are used. Simplify the problem to a small array size and to 16 threads per
block.

• Compile, test, and verify. Increase the array size and retest. When bugs occur, resist the temptation to drop out
of your computer science career and start a new life off-grid without electricity or technology.
Obtain timings. Make sure you remove all printf statements and all cudaDeviceSynchronize() prior to obtaining
this timing (except the last one before the timer records t2). Save this .cu as a separate finished file (mod2.cu) as it’s
needed for your submission.

Modification #3 – Have threads do more than just one swap per iteration

The code up until now has had one thread operating on one pair of values at a time. The goal here is to have each
thread operating on many pairs of values, which in turn better utilizes GPU cache and resources.

• This step requires changing the indexing logic. Suppose you want each GPU thread to work with four pairs of
array indexes instead of one pair. Change your indexing logic so that each CUDA thread computes the same
as before, then increments by some group size, repeats the bitonic logic on new indexes, increments by some
group size, repeats the bitonic logic on new indexes, increments by some group size, and repeats the bitonic
logic on new indexes. 

Said another way, suppose the prior code has a CUDA block has 256 threads and is at a
stage where thread 0 sorts indexes [0] and [1]. Your goal is to keep the CUDA block at 256 threads, but now
have thread 0 sort indexes [0] and [1]. 

Then after all warps have computed their assigned regions, sort indexes
[256] and [257]. Then sorts indexes [512] and [513], and then sorts indexes [768] and [769] accordingly. Avoid
having thread 0 in this example operate on index [0] and [1], then [2] and [3], then [4] and [4], then [6] and [7],
as this results in bad GPU memory coalescing.

• This step requires another loop inside of the GPU code. It will go inside of the j/set GPU loop. (You can choose
to add #pragma unroll X, where X is some integer, to have the compiler manually unroll the loop.)

• The number of CUDA blocks must be reduced to match the number of times this new GPU inner loop iterates.
In other words, if each thread operates on four times as many pairs of values, then four times fewer blocks are
required.

• Determine the new indexing logic. I recommend returning to the spreadsheet here and verifying before
implementing code.

• Start with small threads per block and start with one block. Add in printf and cudaDeviceSynchronize() as
necessary. Test often while increasing the array size and then later the number of blocks.

• Fine tune your code by modifying how many pairs of indexes each thread can work on. Find an optimal value
and use that for your final timing.
Make sure you remove all printf statements and all cudaDeviceSynchronize() prior to obtaining this timing
(except the last one before the timer records t2).
Remember again to test to make sure all values sorted. Also test to make sure all input values came out (ensuring no
input value got duplicated or garbage data got mixed in). For this test, I replaced all floats with unit32_t, then loaded
the array with reverse sorted numbers. I then verified that every index i contains the value i.

Submission:

Submit all .cu code files from which timings were obtained. Prepare a simple report with results. Show the times in
chart with overall speedup from the baseline. Compare times on 2^30 total array values. Explain any relevant
approaches you made beyond these instructions. For example, for Modification #3, I want to know how many pairs of
indexes each CUDA thread operates on.

Here's what I've got so far for my output:

kb24850@morpheus:~$ cd cs6850-sum25

kb24850@morpheus:~/cs6850-sum25$ /usr/local/cuda-12/bin/nvcc -arch=sm_86 bitonic_sort.cu -o bitonic_sort.x

kb24850@morpheus:~/cs6850-sum25$ ./bitonic_sort.x

Progress: 0% done, ETA 4.25 s

Progress: 1% done, ETA 4.23 s

Progress: 2% done, ETA 4.18 s

Progress: 3% done, ETA 4.09 s

Progress: 4% done, ETA 3.97 s

Progress: 5% done, ETA 3.87 s

Progress: 6% done, ETA 3.85 s

Progress: 7% done, ETA 3.74 s

Progress: 8% done, ETA 3.68 s

Progress: 9% done, ETA 3.62 s

Progress: 10% done, ETA 3.56 s

Progress: 11% done, ETA 3.49 s

Progress: 12% done, ETA 3.46 s

Progress: 13% done, ETA 3.38 s

Progress: 14% done, ETA 3.35 s

Progress: 15% done, ETA 3.28 s

Progress: 16% done, ETA 3.23 s

Progress: 17% done, ETA 3.19 s

Progress: 18% done, ETA 3.13 s

Progress: 19% done, ETA 3.09 s

Progress: 20% done, ETA 3.05 s

Progress: 21% done, ETA 2.99 s

Progress: 22% done, ETA 2.95 s

Progress: 23% done, ETA 2.92 s

Progress: 24% done, ETA 2.86 s

Progress: 25% done, ETA 2.81 s

Progress: 26% done, ETA 2.79 s

Progress: 27% done, ETA 2.73 s

Progress: 28% done, ETA 2.68 s

Progress: 29% done, ETA 2.66 s

Progress: 30% done, ETA 2.61 s

Progress: 31% done, ETA 2.56 s

Progress: 32% done, ETA 2.52 s

Progress: 33% done, ETA 2.49 s

Progress: 34% done, ETA 2.44 s

Progress: 35% done, ETA 2.40 s

Progress: 36% done, ETA 2.36 s

Progress: 37% done, ETA 2.32 s

Progress: 38% done, ETA 2.28 s

Progress: 39% done, ETA 2.24 s

Progress: 40% done, ETA 2.20 s

Progress: 41% done, ETA 2.17 s

Progress: 42% done, ETA 2.12 s

Progress: 43% done, ETA 2.09 s

Progress: 44% done, ETA 2.04 s

Progress: 45% done, ETA 2.01 s

Progress: 46% done, ETA 1.97 s

Progress: 47% done, ETA 1.93 s

Progress: 48% done, ETA 1.88 s

Progress: 49% done, ETA 1.85 s

Progress: 50% done, ETA 1.82 s

Progress: 51% done, ETA 1.77 s

Progress: 52% done, ETA 1.74 s

Progress: 53% done, ETA 1.70 s

Progress: 54% done, ETA 1.66 s

Progress: 55% done, ETA 1.63 s

Progress: 56% done, ETA 1.59 s

Progress: 57% done, ETA 1.54 s

Progress: 58% done, ETA 1.51 s

Progress: 59% done, ETA 1.48 s

Progress: 60% done, ETA 1.44 s

Progress: 61% done, ETA 1.40 s

Progress: 62% done, ETA 1.36 s

Progress: 63% done, ETA 1.33 s

Progress: 64% done, ETA 1.29 s

Progress: 65% done, ETA 1.25 s

Progress: 66% done, ETA 1.22 s

Progress: 67% done, ETA 1.18 s

Progress: 68% done, ETA 1.14 s

Progress: 69% done, ETA 1.11 s

Progress: 70% done, ETA 1.07 s

Progress: 71% done, ETA 1.03 s

Progress: 72% done, ETA 1.00 s

Progress: 73% done, ETA 0.96 s

Progress: 74% done, ETA 0.92 s

Progress: 75% done, ETA 0.89 s

Progress: 76% done, ETA 0.85 s

Progress: 77% done, ETA 0.81 s

Progress: 78% done, ETA 0.78 s

Progress: 79% done, ETA 0.74 s

Progress: 80% done, ETA 0.71 s

Progress: 81% done, ETA 0.67 s

Progress: 82% done, ETA 0.64 s

Progress: 83% done, ETA 0.60 s

Progress: 84% done, ETA 0.56 s

Progress: 85% done, ETA 0.53 s

Progress: 86% done, ETA 0.50 s

Progress: 87% done, ETA 0.46 s

Progress: 88% done, ETA 0.42 s

Progress: 89% done, ETA 0.39 s

Progress: 90% done, ETA 0.35 s

Progress: 91% done, ETA 0.31 s

Progress: 92% done, ETA 0.28 s

Progress: 93% done, ETA 0.24 s

Progress: 94% done, ETA 0.20 s

Progress: 95% done, ETA 0.17 s

Progress: 96% done, ETA 0.14 s

Progress: 97% done, ETA 0.10 s

Progress: 98% done, ETA 0.07 s

Progress: 99% done, ETA 0.03 s

Progress: 100% done, ETA 0.00 s

Kernel wall time elapsed: 3506.85 ms

Array sorted correctly

kb24850@morpheus:~/cs6850-sum25$ /usr/local/cuda-12/bin/nvcc -arch=sm_86 mod1.cu -o mod1.x

kb24850@morpheus:~/cs6850-sum25$ ./mod1.x

Progress: 0% done, ETA 4.25 s

Progress: 1% done, ETA 4.24 s

Progress: 2% done, ETA 4.19 s

Progress: 3% done, ETA 4.10 s

Progress: 4% done, ETA 3.98 s

Progress: 5% done, ETA 3.88 s

Progress: 6% done, ETA 3.86 s

Progress: 7% done, ETA 3.75 s

Progress: 8% done, ETA 3.68 s

Progress: 9% done, ETA 3.62 s

Progress: 10% done, ETA 3.57 s

Progress: 11% done, ETA 3.50 s

Progress: 12% done, ETA 3.47 s

Progress: 13% done, ETA 3.38 s

Progress: 14% done, ETA 3.36 s

Progress: 15% done, ETA 3.29 s

Progress: 16% done, ETA 3.24 s

Progress: 17% done, ETA 3.19 s

Progress: 18% done, ETA 3.14 s

Progress: 19% done, ETA 3.10 s

Progress: 20% done, ETA 3.06 s

Progress: 21% done, ETA 3.00 s

Progress: 22% done, ETA 2.96 s

Progress: 23% done, ETA 2.92 s

Progress: 24% done, ETA 2.86 s

Progress: 25% done, ETA 2.82 s

Progress: 26% done, ETA 2.79 s

Progress: 27% done, ETA 2.74 s

Progress: 28% done, ETA 2.69 s

Progress: 29% done, ETA 2.66 s

Progress: 30% done, ETA 2.61 s

Progress: 31% done, ETA 2.56 s

Progress: 32% done, ETA 2.53 s

Progress: 33% done, ETA 2.49 s

Progress: 34% done, ETA 2.44 s

Progress: 35% done, ETA 2.40 s

Progress: 36% done, ETA 2.36 s

Progress: 37% done, ETA 2.33 s

Progress: 38% done, ETA 2.29 s

Progress: 39% done, ETA 2.24 s

Progress: 40% done, ETA 2.21 s

Progress: 41% done, ETA 2.17 s

Progress: 42% done, ETA 2.13 s

Progress: 43% done, ETA 2.09 s

Progress: 44% done, ETA 2.05 s

Progress: 45% done, ETA 2.01 s

Progress: 46% done, ETA 1.98 s

Progress: 47% done, ETA 1.93 s

Progress: 48% done, ETA 1.89 s

Progress: 49% done, ETA 1.86 s

Progress: 50% done, ETA 1.82 s

Progress: 51% done, ETA 1.78 s

Progress: 52% done, ETA 1.74 s

Progress: 53% done, ETA 1.70 s

Progress: 54% done, ETA 1.66 s

Progress: 55% done, ETA 1.63 s

Progress: 56% done, ETA 1.59 s

Progress: 57% done, ETA 1.55 s

Progress: 58% done, ETA 1.52 s

Progress: 59% done, ETA 1.48 s

Progress: 60% done, ETA 1.45 s

Progress: 61% done, ETA 1.40 s

Progress: 62% done, ETA 1.36 s

Progress: 63% done, ETA 1.33 s

Progress: 64% done, ETA 1.29 s

Progress: 65% done, ETA 1.25 s

Progress: 66% done, ETA 1.22 s

Progress: 67% done, ETA 1.18 s

Progress: 68% done, ETA 1.14 s

Progress: 69% done, ETA 1.11 s

Progress: 70% done, ETA 1.07 s

Progress: 71% done, ETA 1.03 s

Progress: 72% done, ETA 1.00 s

Progress: 73% done, ETA 0.96 s

Progress: 74% done, ETA 0.92 s

Progress: 75% done, ETA 0.89 s

Progress: 76% done, ETA 0.85 s

Progress: 77% done, ETA 0.81 s

Progress: 78% done, ETA 0.78 s

Progress: 79% done, ETA 0.74 s

Progress: 80% done, ETA 0.71 s

Progress: 81% done, ETA 0.67 s

Progress: 82% done, ETA 0.63 s

Progress: 83% done, ETA 0.60 s

Progress: 84% done, ETA 0.56 s

Progress: 85% done, ETA 0.53 s

Progress: 86% done, ETA 0.49 s

Progress: 87% done, ETA 0.46 s

Progress: 88% done, ETA 0.42 s

Progress: 89% done, ETA 0.39 s

Progress: 90% done, ETA 0.35 s

Progress: 91% done, ETA 0.31 s

Progress: 92% done, ETA 0.28 s

Progress: 93% done, ETA 0.24 s

Progress: 94% done, ETA 0.20 s

Progress: 95% done, ETA 0.17 s

Progress: 96% done, ETA 0.14 s

Progress: 97% done, ETA 0.10 s

Progress: 98% done, ETA 0.07 s

Progress: 99% done, ETA 0.03 s

Progress: 100% done, ETA 0.00 s

Kernel wall time elapsed: 3500.08 ms

Array sorted correctly

kb24850@morpheus:~/cs6850-sum25$ /usr/local/cuda-12/bin/nvcc -arch=sm_86 mod2.cu -o mod2.x

kb24850@morpheus:~/cs6850-sum25$ ./mod2.x

Progress: 0% done, ETA 4.26 s

Progress: 1% done, ETA 2.13 s

Progress: 2% done, ETA 1.69 s

Progress: 3% done, ETA 1.40 s

Progress: 4% done, ETA 1.22 s

Progress: 5% done, ETA 1.09 s

Progress: 6% done, ETA 1.09 s

Progress: 7% done, ETA 0.99 s

Progress: 8% done, ETA 0.90 s

Progress: 9% done, ETA 0.90 s

Progress: 10% done, ETA 0.88 s

Progress: 11% done, ETA 0.88 s

Progress: 12% done, ETA 0.91 s

Progress: 13% done, ETA 0.88 s

Progress: 14% done, ETA 0.88 s

Progress: 15% done, ETA 0.90 s

Progress: 16% done, ETA 0.90 s

Progress: 17% done, ETA 0.94 s

Progress: 18% done, ETA 0.92 s

Progress: 19% done, ETA 0.92 s

Progress: 20% done, ETA 0.95 s

Progress: 21% done, ETA 0.93 s

Progress: 22% done, ETA 0.93 s

Progress: 23% done, ETA 0.96 s

Progress: 24% done, ETA 0.94 s

Progress: 25% done, ETA 0.94 s

Progress: 26% done, ETA 0.95 s

Progress: 27% done, ETA 0.99 s

Progress: 28% done, ETA 0.94 s

Progress: 29% done, ETA 0.94 s

Progress: 30% done, ETA 0.97 s

Progress: 31% done, ETA 0.94 s

Progress: 32% done, ETA 0.94 s

Progress: 33% done, ETA 0.94 s

Progress: 34% done, ETA 0.97 s

Progress: 35% done, ETA 0.92 s

Progress: 36% done, ETA 0.92 s

Progress: 37% done, ETA 0.93 s

Progress: 38% done, ETA 0.94 s

Progress: 39% done, ETA 0.90 s

Progress: 40% done, ETA 0.90 s

Progress: 41% done, ETA 0.90 s

Progress: 42% done, ETA 0.91 s

Progress: 43% done, ETA 0.91 s

Progress: 44% done, ETA 0.86 s

Progress: 45% done, ETA 0.86 s

Progress: 46% done, ETA 0.87 s

Progress: 47% done, ETA 0.87 s

Progress: 48% done, ETA 0.82 s

Progress: 49% done, ETA 0.82 s

Progress: 50% done, ETA 0.82 s

Progress: 51% done, ETA 0.82 s

Progress: 52% done, ETA 0.82 s

Progress: 53% done, ETA 0.76 s

Progress: 54% done, ETA 0.76 s

Progress: 55% done, ETA 0.76 s

Progress: 56% done, ETA 0.76 s

Progress: 57% done, ETA 0.75 s

Progress: 58% done, ETA 0.70 s

Progress: 59% done, ETA 0.70 s

Progress: 60% done, ETA 0.70 s

Progress: 61% done, ETA 0.69 s

Progress: 62% done, ETA 0.68 s

Progress: 63% done, ETA 0.63 s

Progress: 64% done, ETA 0.63 s

Progress: 65% done, ETA 0.62 s

Progress: 66% done, ETA 0.61 s

Progress: 67% done, ETA 0.60 s

Progress: 68% done, ETA 0.55 s

Progress: 69% done, ETA 0.55 s

Progress: 70% done, ETA 0.54 s

Progress: 71% done, ETA 0.53 s

Progress: 72% done, ETA 0.52 s

Progress: 73% done, ETA 0.50 s

Progress: 74% done, ETA 0.45 s

Progress: 75% done, ETA 0.45 s

Progress: 76% done, ETA 0.44 s

Progress: 77% done, ETA 0.43 s

Progress: 78% done, ETA 0.42 s

Progress: 79% done, ETA 0.40 s

Progress: 80% done, ETA 0.35 s

Progress: 81% done, ETA 0.35 s

Progress: 82% done, ETA 0.34 s

Progress: 83% done, ETA 0.33 s

Progress: 84% done, ETA 0.31 s

Progress: 85% done, ETA 0.29 s

Progress: 86% done, ETA 0.24 s

Progress: 87% done, ETA 0.24 s

Progress: 88% done, ETA 0.23 s

Progress: 89% done, ETA 0.21 s

Progress: 90% done, ETA 0.19 s

Progress: 91% done, ETA 0.17 s

Progress: 92% done, ETA 0.13 s

Progress: 93% done, ETA 0.13 s

Progress: 94% done, ETA 0.11 s

Progress: 95% done, ETA 0.10 s

Progress: 96% done, ETA 0.08 s

Progress: 97% done, ETA 0.06 s

Progress: 98% done, ETA 0.04 s

Progress: 99% done, ETA 0.00 s

Progress: 100% done, ETA 0.00 s

Kernel wall time elapsed: 1979.59 ms

Copy: 0% done, ETA 0.87 s

Copy: 1% done, ETA 0.87 s

Copy: 2% done, ETA 0.86 s

Copy: 3% done, ETA 0.86 s

Copy: 4% done, ETA 0.85 s

Copy: 5% done, ETA 0.83 s

Copy: 6% done, ETA 0.83 s

Copy: 7% done, ETA 0.82 s

Copy: 8% done, ETA 0.81 s

Copy: 9% done, ETA 0.81 s

Copy: 10% done, ETA 0.79 s

Copy: 11% done, ETA 0.78 s

Copy: 12% done, ETA 0.78 s

Copy: 13% done, ETA 0.76 s

Copy: 14% done, ETA 0.76 s

Copy: 15% done, ETA 0.75 s

Copy: 16% done, ETA 0.74 s

Copy: 17% done, ETA 0.74 s

Copy: 18% done, ETA 0.72 s

Copy: 19% done, ETA 0.71 s

Copy: 20% done, ETA 0.71 s

Copy: 21% done, ETA 0.70 s

Copy: 22% done, ETA 0.68 s

Copy: 23% done, ETA 0.68 s

Copy: 24% done, ETA 0.67 s

Copy: 25% done, ETA 0.67 s

Copy: 26% done, ETA 0.65 s

Copy: 27% done, ETA 0.64 s

Copy: 28% done, ETA 0.64 s

Copy: 29% done, ETA 0.63 s

Copy: 30% done, ETA 0.61 s

Copy: 31% done, ETA 0.61 s

Copy: 32% done, ETA 0.60 s

Copy: 33% done, ETA 0.58 s

Copy: 34% done, ETA 0.58 s

Copy: 35% done, ETA 0.57 s

Copy: 36% done, ETA 0.56 s

Copy: 37% done, ETA 0.56 s

Copy: 38% done, ETA 0.54 s

Copy: 39% done, ETA 0.54 s

Copy: 40% done, ETA 0.53 s

Copy: 41% done, ETA 0.51 s

Copy: 42% done, ETA 0.51 s

Copy: 43% done, ETA 0.50 s

Copy: 44% done, ETA 0.49 s

Copy: 45% done, ETA 0.49 s

Copy: 46% done, ETA 0.47 s

Copy: 47% done, ETA 0.46 s

Copy: 48% done, ETA 0.46 s

Copy: 49% done, ETA 0.44 s

Copy: 50% done, ETA 0.44 s

Copy: 51% done, ETA 0.43 s

Copy: 52% done, ETA 0.42 s

Copy: 53% done, ETA 0.42 s

Copy: 54% done, ETA 0.40 s

Copy: 55% done, ETA 0.39 s

Copy: 56% done, ETA 0.39 s

Copy: 57% done, ETA 0.38 s

Copy: 58% done, ETA 0.36 s

Copy: 59% done, ETA 0.36 s

Copy: 60% done, ETA 0.35 s

Copy: 61% done, ETA 0.33 s

Copy: 62% done, ETA 0.33 s

Copy: 63% done, ETA 0.32 s

Copy: 64% done, ETA 0.32 s

Copy: 65% done, ETA 0.31 s

Copy: 66% done, ETA 0.29 s

Copy: 67% done, ETA 0.29 s

Copy: 68% done, ETA 0.28 s

Copy: 69% done, ETA 0.26 s

Copy: 70% done, ETA 0.26 s

Copy: 71% done, ETA 0.25 s

Copy: 72% done, ETA 0.24 s

Copy: 73% done, ETA 0.24 s

Copy: 74% done, ETA 0.22 s

Copy: 75% done, ETA 0.22 s

Copy: 76% done, ETA 0.21 s

Copy: 77% done, ETA 0.19 s

Copy: 78% done, ETA 0.19 s

Copy: 79% done, ETA 0.18 s

Copy: 80% done, ETA 0.17 s

Copy: 81% done, ETA 0.17 s

Copy: 82% done, ETA 0.15 s

Copy: 83% done, ETA 0.14 s

Copy: 84% done, ETA 0.14 s

Copy: 85% done, ETA 0.13 s

Copy: 86% done, ETA 0.11 s

Copy: 87% done, ETA 0.11 s

Copy: 88% done, ETA 0.10 s

Copy: 89% done, ETA 0.10 s

Copy: 90% done, ETA 0.08 s

Copy: 91% done, ETA 0.07 s

Copy: 92% done, ETA 0.07 s

Copy: 93% done, ETA 0.06 s

Copy: 94% done, ETA 0.04 s

Copy: 95% done, ETA 0.04 s

Copy: 96% done, ETA 0.03 s

Copy: 97% done, ETA 0.01 s

Copy: 98% done, ETA 0.01 s

Copy: 99% done, ETA 0.00 s

Copy: 100% done, ETA 0.00 s

Verify: 0% done, ETA 149.25 s

Verify: 1% done, ETA 10.60 s

Verify: 2% done, ETA 10.68 s

Verify: 3% done, ETA 10.63 s

Verify: 4% done, ETA 10.54 s

Verify: 5% done, ETA 10.45 s

Verify: 6% done, ETA 10.35 s

Verify: 7% done, ETA 10.11 s

Verify: 8% done, ETA 9.88 s

Verify: 9% done, ETA 9.67 s

Verify: 10% done, ETA 9.48 s

Verify: 11% done, ETA 9.31 s

Verify: 12% done, ETA 9.15 s

Verify: 13% done, ETA 9.01 s

Verify: 14% done, ETA 8.88 s

Verify: 15% done, ETA 8.76 s

Verify: 16% done, ETA 8.62 s

Verify: 17% done, ETA 8.49 s

Verify: 18% done, ETA 8.37 s

Verify: 19% done, ETA 8.25 s

Verify: 20% done, ETA 8.13 s

Verify: 21% done, ETA 8.01 s

Verify: 22% done, ETA 7.91 s

Verify: 23% done, ETA 7.80 s

Verify: 24% done, ETA 7.69 s

Verify: 25% done, ETA 7.58 s

Verify: 26% done, ETA 7.47 s

Verify: 27% done, ETA 7.36 s

Verify: 28% done, ETA 7.25 s

Verify: 29% done, ETA 7.14 s

Verify: 30% done, ETA 7.03 s

Verify: 31% done, ETA 6.92 s

Verify: 32% done, ETA 6.82 s

Verify: 33% done, ETA 6.72 s

Verify: 34% done, ETA 6.61 s

Verify: 35% done, ETA 6.50 s

Verify: 36% done, ETA 6.40 s

Verify: 37% done, ETA 6.29 s

Verify: 38% done, ETA 6.19 s

Verify: 39% done, ETA 6.09 s

Verify: 40% done, ETA 5.98 s

Verify: 41% done, ETA 5.88 s

Verify: 42% done, ETA 5.78 s

Verify: 43% done, ETA 5.67 s

Verify: 44% done, ETA 5.58 s

Verify: 45% done, ETA 5.48 s

Verify: 46% done, ETA 5.37 s

Verify: 47% done, ETA 5.27 s

Verify: 48% done, ETA 5.17 s

Verify: 49% done, ETA 5.07 s

Verify: 50% done, ETA 4.97 s

Verify: 51% done, ETA 4.87 s

Verify: 52% done, ETA 4.76 s

Verify: 53% done, ETA 4.66 s

Verify: 54% done, ETA 4.56 s

Verify: 55% done, ETA 4.46 s

Verify: 56% done, ETA 4.36 s

Verify: 57% done, ETA 4.26 s

Verify: 58% done, ETA 4.16 s

Verify: 59% done, ETA 4.06 s

Verify: 60% done, ETA 3.96 s

Verify: 61% done, ETA 3.86 s

Verify: 62% done, ETA 3.76 s

Verify: 63% done, ETA 3.66 s

Verify: 64% done, ETA 3.56 s

Verify: 65% done, ETA 3.46 s

Verify: 66% done, ETA 3.36 s

Verify: 67% done, ETA 3.26 s

Verify: 68% done, ETA 3.17 s

Verify: 69% done, ETA 3.07 s

Verify: 70% done, ETA 2.97 s

Verify: 71% done, ETA 2.87 s

Verify: 72% done, ETA 2.77 s

Verify: 73% done, ETA 2.67 s

Verify: 74% done, ETA 2.57 s

Verify: 75% done, ETA 2.47 s

Verify: 76% done, ETA 2.37 s

Verify: 77% done, ETA 2.27 s

Verify: 78% done, ETA 2.17 s

Verify: 79% done, ETA 2.07 s

Verify: 80% done, ETA 1.97 s

Verify: 81% done, ETA 1.88 s

Verify: 82% done, ETA 1.78 s

Verify: 83% done, ETA 1.68 s

Verify: 84% done, ETA 1.58 s

Verify: 85% done, ETA 1.48 s

Verify: 86% done, ETA 1.38 s

Verify: 87% done, ETA 1.28 s

Verify: 88% done, ETA 1.18 s

Verify: 89% done, ETA 1.09 s

Verify: 90% done, ETA 0.99 s

Verify: 91% done, ETA 0.89 s

Verify: 92% done, ETA 0.79 s

Verify: 93% done, ETA 0.69 s

Verify: 94% done, ETA 0.59 s

Verify: 95% done, ETA 0.49 s

Verify: 96% done, ETA 0.39 s

Verify: 97% done, ETA 0.30 s

Verify: 98% done, ETA 0.20 s

Verify: 99% done, ETA 0.10 s

Verify: 100% done, ETA 0.00 s

Array sorted correctly

kb24850@morpheus:~/cs6850-sum25$ /usr/local/cuda-12/bin/nvcc -arch=sm_86 mod3.cu -o mod3.x

kb24850@morpheus:~/cs6850-sum25$ ./mod3.x

Progress: 0% done, ETA 4.27 s

Progress: 1% done, ETA 4.26 s

Progress: 2% done, ETA 4.20 s

Progress: 3% done, ETA 4.12 s

Progress: 4% done, ETA 4.00 s

Progress: 5% done, ETA 3.90 s

Progress: 6% done, ETA 3.88 s

Progress: 7% done, ETA 3.77 s

Progress: 8% done, ETA 3.70 s

Progress: 9% done, ETA 3.65 s

Progress: 10% done, ETA 3.59 s

Progress: 11% done, ETA 3.52 s

Progress: 12% done, ETA 3.49 s

Progress: 13% done, ETA 3.41 s

Progress: 14% done, ETA 3.38 s

Progress: 15% done, ETA 3.31 s

Progress: 16% done, ETA 3.26 s

Progress: 17% done, ETA 3.21 s

Progress: 18% done, ETA 3.16 s

Progress: 19% done, ETA 3.12 s

Progress: 20% done, ETA 3.08 s

Progress: 21% done, ETA 3.02 s

Progress: 22% done, ETA 2.98 s

Progress: 23% done, ETA 2.94 s

Progress: 24% done, ETA 2.88 s

Progress: 25% done, ETA 2.84 s

Progress: 26% done, ETA 2.81 s

Progress: 27% done, ETA 2.76 s

Progress: 28% done, ETA 2.71 s

Progress: 29% done, ETA 2.68 s

Progress: 30% done, ETA 2.63 s

Progress: 31% done, ETA 2.58 s

Progress: 32% done, ETA 2.54 s

Progress: 33% done, ETA 2.51 s

Progress: 34% done, ETA 2.46 s

Progress: 35% done, ETA 2.42 s

Progress: 36% done, ETA 2.38 s

Progress: 37% done, ETA 2.34 s

Progress: 38% done, ETA 2.30 s

Progress: 39% done, ETA 2.26 s

Progress: 40% done, ETA 2.22 s

Progress: 41% done, ETA 2.19 s

Progress: 42% done, ETA 2.14 s

Progress: 43% done, ETA 2.10 s

Progress: 44% done, ETA 2.06 s

Progress: 45% done, ETA 2.03 s

Progress: 46% done, ETA 1.99 s

Progress: 47% done, ETA 1.95 s

Progress: 48% done, ETA 1.90 s

Progress: 49% done, ETA 1.87 s

Progress: 50% done, ETA 1.83 s

Progress: 51% done, ETA 1.79 s

Progress: 52% done, ETA 1.75 s

Progress: 53% done, ETA 1.71 s

Progress: 54% done, ETA 1.68 s

Progress: 55% done, ETA 1.64 s

Progress: 56% done, ETA 1.60 s

Progress: 57% done, ETA 1.56 s

Progress: 58% done, ETA 1.53 s

Progress: 59% done, ETA 1.49 s

Progress: 60% done, ETA 1.46 s

Progress: 61% done, ETA 1.41 s

Progress: 62% done, ETA 1.37 s

Progress: 63% done, ETA 1.34 s

Progress: 64% done, ETA 1.30 s

Progress: 65% done, ETA 1.26 s

Progress: 66% done, ETA 1.23 s

Progress: 67% done, ETA 1.19 s

Progress: 68% done, ETA 1.15 s

Progress: 69% done, ETA 1.12 s

Progress: 70% done, ETA 1.08 s

Progress: 71% done, ETA 1.04 s

Progress: 72% done, ETA 1.01 s

Progress: 73% done, ETA 0.97 s

Progress: 74% done, ETA 0.93 s

Progress: 75% done, ETA 0.90 s

Progress: 76% done, ETA 0.86 s

Progress: 77% done, ETA 0.82 s

Progress: 78% done, ETA 0.79 s

Progress: 79% done, ETA 0.75 s

Progress: 80% done, ETA 0.72 s

Progress: 81% done, ETA 0.68 s

Progress: 82% done, ETA 0.64 s

Progress: 83% done, ETA 0.61 s

Progress: 84% done, ETA 0.57 s

Progress: 85% done, ETA 0.53 s

Progress: 86% done, ETA 0.50 s

Progress: 87% done, ETA 0.46 s

Progress: 88% done, ETA 0.42 s

Progress: 89% done, ETA 0.39 s

Progress: 90% done, ETA 0.35 s

Progress: 91% done, ETA 0.31 s

Progress: 92% done, ETA 0.28 s

Progress: 93% done, ETA 0.24 s

Progress: 94% done, ETA 0.21 s

Progress: 95% done, ETA 0.18 s

Progress: 96% done, ETA 0.14 s

Progress: 97% done, ETA 0.10 s

Progress: 98% done, ETA 0.07 s

Progress: 99% done, ETA 0.03 s

Progress: 100% done, ETA 0.00 s

Kernel wall time elapsed: 3527.69 ms

Array sorted correctly

kb24850@morpheus:~/cs6850-sum25$ 




--------------------------------------------------------------------------------------------------------------------------
== We're Using GitHub Under Protest ==

This project is currently hosted on GitHub.  This is not ideal; GitHub is a
proprietary, trade-secret system that is not Free and Open Souce Software
(FOSS).  We are deeply concerned about using a proprietary system like GitHub
to develop our FOSS project. I have a [website](https://bellKevin.me) where the
project contributors are actively discussing how we can move away from GitHub
in the long term.  We urge you to read about the [Give up GitHub](https://GiveUpGitHub.org) campaign 
from [the Software Freedom Conservancy](https://sfconservancy.org) to understand some of the reasons why GitHub is not 
a good place to host FOSS projects.

If you are a contributor who personally has already quit using GitHub, please
email me at **bellKevin@pm.me** for how to send us contributions without
using GitHub directly.

Any use of this project's code by GitHub Copilot, past or present, is done
without our permission.  We do not consent to GitHub's use of this project's
code in Copilot.

![Logo of the GiveUpGitHub campaign](https://sfconservancy.org/img/GiveUpGitHub.png)

<p align="right"><a href="#readme-top">back to top</a></p>
