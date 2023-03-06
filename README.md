# HPC Homework 2

Yiwei Shao

**Experiment Machine:** 

Processor: AMD EPYC 7452 32-Core Processor 1.5GHz
g++ version:11.3.0

Architecture: x86_64

# Problem 2

The outputs of different optimization flags are in directory *problem2/opt*  

The outputs of different *BLOCK_SIZE* are in directory *problem2/block*

When the program is optimized with *O0*, *O1* and *O2*, the block version is always about 2 times faster than the original version, which matches our expectation. 

But when switching to O3 optimization flag, the original version is faster than the block version. More specifically, the running time of the block version does not reduce while the original version speedup about 4X. According to [https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html), the *O3* optimization does more loop optimization than *O2.* That might be the reason since the original version only contains simple loops and is easier for the compiler to optimize.

The running time of both versions will increase dramatically after dimension 1024. That might because L3 machine is 16Mb and it can accommodate $1024\times1024$ double precision floats at most.

When changing the *BLOCK_SIZE* with optimization flag O3, the cost time for the biggest matrix decreases until *BLOCK_SIZE*=32. So 32 is an optimal value for matrix multiplication.

# Problem 4

### (a)

The inner product is implemented in *inner_prod.cpp.*

- compute_fn01()  Naive algorithm.
- compute_fn02()  Unroll 2, pipeline
- compute_fn03()  Unroll 2, pipeline with index optimization
- compute_fn04()  Unroll 2, pipeline with index optimization and disentangle
- compute_fn05()  Unroll 4, pipeline with index optimization and disentangle

The outputs of different implementations are in directory *Problem4*

According to the output files, the solving time increases dramatically when vector size increases to $524288$. And the total sizes of these two vectors are 8Mb. Considering the L3 cache for my machine is 16 Mb and my program needs a little more cache than the two vectors take up, the best vector size should be slightly smaller than $2\times524288$.

### (b)

- **compute.cpp**
    
    **multiply-add**
    
    | Optimization Flag | Running Time (s) | Cycles per Evaluation | GFLOPS |
    | --- | --- | --- | --- |
    | O3 | 1.497518 | 2.246359 | 1.335493 |
    | O2 | 1.497929  | 2.246976 | 1.335126  |
    | O1 | 3.891597 | 5.837482 | 0.513920 |
    | O0 | 4.248807 | 6.373301 | 0.470714 |
    
    **division**
    
    | Optimization Flag | Running Time (s) | Cycles per Evaluation | GFLOPS |
    | --- | --- | --- | --- |
    | O3 | 3.903214 | 5.854896 | 0.512392 |
    | O2 | 3.894030 | 5.841128 | 0.513599 |
    | O1 | 6.000244 | 9.000456 | 0.333316 |
    | O0 | 6.313590 | 9.470529 | 0.316772 |
    
    **sqrt**
    
    | Optimization Flag | Running Time (s) | Cycles per Evaluation | GFLOPS |
    | --- | --- | --- | --- |
    | O3 | 6.004123 | 9.006265 | 0.333101 |
    | O2 | 6.010781 | 9.016312 | 0.332730 |
    | O1 | 8.402271 | 12.603510 | 0.238029 |
    | O0 | 10.192865 | 15.289367 | 0.196215 |
    
    **sin**
    
    | Optimization Flag | Running Time (s) | Cycles per Evaluation | GFLOPS |
    | --- | --- | --- | --- |
    | O3 | 11.656103 | 17.484359 | 0.171582 |
    | O2 | 11.652059 | 17.478183 | 0.171643 |
    | O1 | 13.251104 | 19.876739 | 0.150930 |
    | O0 | 15.095802 | 22.643796 | 0.132487 |
- **compute-vec.cpp**

Output with OpenMP

```cpp
time = 1.510941
flop-rate = 5.294521 Gflop/s

time = 1.497440
flop-rate = 5.342419 Gflop/s

time = 1.498995
flop-rate = 5.336869 Gflop/s
```

Output with **#pragma unroll**

```cpp
compute-vec.cpp:16:21: optimized: loop vectorized using 32 byte vectors
compute-vec.cpp:16:21: optimized:  loop versioned for vectorization because of possible aliasing
compute-vec.cpp:52:21: optimized: loop vectorized using 16 byte vectors
compute-vec.cpp:46:5: optimized: basic block part vectorized using 32 byte vectors
time = 1.499241
flop-rate = 5.335839 Gflop/s

time = 1.498447
flop-rate = 5.338831 Gflop/s

time = 1.498376
flop-rate = 5.339084 Gflop/s
```

Output with **#pragma GCC ivdep**

```cpp
compute-vec.cpp:16:17: optimized: loop vectorized using 32 byte vectors
compute-vec.cpp:52:21: optimized: loop vectorized using 16 byte vectors
compute-vec.cpp:46:5: optimized: basic block part vectorized using 32 byte vectors
time = 1.501182
flop-rate = 5.328966 Gflop/s

time = 1.497545
flop-rate = 5.342031 Gflop/s

time = 1.499832
flop-rate = 5.333898 Gflop/s
```

**#pragma unroll** let compiler decide if unroll the loop and **#pragma GCC ivdep** force the compiler to ignore the loop dependency. The optimized information output shows that the compiler successfully unrolled the loop and they gained similar improvement for this problem. **OpenMP** does the similar optimization with **AVX**

- **compute-vec-pipe.cpp**

Running time for different **M** and **functions**

**OpenMP**

| M | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| fn0 | 1.498873 | 1.513063 | 1.513183 | 2.861379 | 2.949068 | 11.432868 |
| fn1 | 1.501146 | 1.505750 | 1.497094 | 2.399404 | 5.425581 | 10.017032 |
| fn2 | 1.498717 | 1.500135 | 1.497746 | 2.398358 | 5.448774 | 9.979667 |

**#pragma unroll**

| M | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| fn0 | 1.512504 | 1.497755 | 1.511739 | 1.512873 | 2.868334 | 10.196745 |
| fn1 | 1.499327 | 1.496293 | 1.500039 | 1.512967 | 5.201772 | 10.215604 |
| fn2 | 1.498232 | 1.496820 | 1.499235 | 1.518770 | 5.221702 | 10.359698 |

**#pragma GCC ivdep**

| M | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| fn0 | 1.510875 | 1.499915 | 1.500936 | 1.512461 | 2.852350 | 10.292027 |
| fn1 | 1.497899 | 1.499319 | 1.497990 | 1.543855 | 5.195912 | 10.216272 |
| fn2 | 1.500909 | 1.499360 | 1.497704 | 1.513994 | 5.303215 | 10.365627 |

As we can see the running time increases only after M is bigger than 8, and that might because the **AVX** vector is packed of 4 and there are 2 **FMA** for each core. So we can do 8 multiply-add one time. When vector size is bigger than 8, we need to call more **AVX** functions and the cost time is no longer constant. **OpenMP** takes much more time when vector size is 8. That might because **OpenMP** cannot fully utilize the 2 **FMA** since it is cross-platform.