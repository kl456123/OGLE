# OGLE
opengl2.0 based deeplearning inference engine, used for stricted environment,
like web, opencl-disable arm


## Introduce
OGLE is mainly for deeplearning inferece, it has two parts,
1. Model Converter
2. opengl based graph runtime
you need to convert model you trained to onnx format first. 
then use converter to convert model from onnx to dlx format, only dlx
model can used for graph runtime

## Install
### prerequirements
1. opengl libraries
2. common utils libraries, like glog gtest gflag
3. opencv, protobuf please build from source


# compile
```bash
# build converter
cd converter
mkdir build && cd build && cmake .. && make -j`nproco`

# build runtime
cd ..
cd ./ogl_runtime/opengl/nn
# generate kernel files
python make_shaders.py glsl all_shaders.h all_shaders.cc
cd -

mkdir build && cd build && cmake .. && make -j`nproco`
```
## How to use
* convert model first
```bash
cd converter/build
export SRC=model.onnx
exoprt DST=demo.dlx
./Converter ${SRC} ${DST}
# then you can get the demo.dlx in current directory
```

* to use opengl based graph runtime, please refer to
`ogl_runtime/opengl/examples/ssd/main.cc` for example.

```bash
# demo
cd ogl_runtime/build
./opengl/examples/ssd_detector
```

## Benchmark
| framework | model | time | platform |
| OGLE | mobilenet ssd | 8ms | 1080TI |
| MNN | mobilenet ssd | 9-10ms | 1080TI |

## Develop

1. add new op
please read `ogl_runtime/opengl/nn/glsl` and
`ogl_runtime/opengl/nn/kernels` for reference. then you need to rerun
`python make_shaders.py glsl all_shaders.h all_shaders.cc` again