# Runtime Profiler

collect statistic time for each stage during execution to help
to find bottleneck, improve the performance and reduce the
latency of the pipeline



## Pipeline

1. image preprocess in cpu
2. data transfer from cpu to gpu
3. data copy in gpu
4. launch kernel
5. actual kernel computation
6. data transfer from gpu to cpu
7. image postprocess in cpu

## Timer

1. Timer in host

2. Timer in device


## Tips
1. use DMA to do data transfer
2. do preprocess and postprocess using gpu
or cpu multiple threads
3. data memory manager(memory pool, bfc)
