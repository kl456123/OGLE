# Kernel Optimizer

* After graph optimizer, we need to generate node call api and kernel code
for actual computation. it is necessary to know more information about device,
dispatch node to different compute kernel according to node attributes(like kernel size in conv2d)

## TODO
1. optimize multiple for loops
2. use device manager to search device infos
3. try different dformats like hwIO4
4. autotune work size

