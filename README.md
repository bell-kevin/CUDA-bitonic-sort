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
A fairly straightforward description of a bitonic sort can be given in both text and in pictures. The single threaded CPU
implementation is also fairly clean and straightforward. Consider the following figure:
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

 Bitonic Sort on CUDA. On a quick benchmark it was 10x faster than the CPU version. 

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

To compile we'll use:

/usr/local/cuda-12/bin/nvcc -arch=sm_86 bitonic_sort.cu -o bitonic_sort.x

After compiling, you can run the program with:

./bitonic_sort.x

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
• Once the new spreadsheet columns demonstrate proper logic, translate this into CUDA/C++ code. Update
accordingly the printf statements for debugging/comparison.
• Update the kernel invocation to account for the fact that somehow you need only half as many threads as
before (or said another way, each kernel can work with twice as many index values).
• Test your code and ensure it sorts. Carefully dial up the array size by increasing powers of two. Test and
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
logic on new indexes. Said another way, suppose the prior code has a CUDA block has 256 threads and is at a
stage where thread 0 sorts indexes [0] and [1]. Your goal is to keep the CUDA block at 256 threads, but now
have thread 0 sort indexes [0] and [1]. Then after all warps have computed their assigned regions, sort indexes
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
