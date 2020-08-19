# OGLE
opengl2.0 based deeplearning inference engine, used for stricted environment,
like web, opencl-disable arm


## Introduction
OGLE is mainly for deeplearning inferece, it has two parts,
1. Model Converter
2. opengl based graph runtime
you need to convert model you trained to onnx format first.
then use converter to convert model from onnx to dlx format, only dlx
model can used for graph runtime

## Install
### Prerequisite
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

* trained model
download from [google drive](https://drive.google.com/file/d/1yLALDmJ7F1FoePrkPVCY0l5cFlvsZ0X5/view?usp=sharing)
or [baiduyunpan](https://pan.baidu.com/s/18Mii22nKlZLhaN-IgBjjQA) , passwd: ef7i, then put `demo.dlx`
into build directory, you can use the person detector model to detect in camera or image. the follow image is used
for demostration.
![](https://github.com/kl456123/OGLE/blob/master/ogl_runtime/opengl/examples/ssd/result.jpg)

## Benchmark

| framework | model | time | platform | mem |
| -----| ---- | ---- | --- | --- |
| OGLE | mobilenet ssd | 8ms | 1080TI | 187m |
| MNN | mobilenet ssd | 9-10ms | 1080TI | 174m |

## Develop

1. add new op
please read `ogl_runtime/opengl/nn/glsl` and
`ogl_runtime/opengl/nn/kernels` for reference. then you need to rerun
`python make_shaders.py glsl all_shaders.h all_shaders.cc` again
