#include "opengl/nn/kernels/shape.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"


namespace opengl{
    ShapeKernel::ShapeKernel(Context* context)
        :Kernel(context){
            // kernel_fname_ = "../opengl/nn/glsl/shape.glsl";
        }

    void ShapeKernel::SetupAttr(const dlxnet::Attribute& attr){
    }

    void ShapeKernel::Compute(TensorList& inputs, TensorList& outputs){
        DLOG(INFO)<<"ShapeKernel Inputs: "<<inputs.size();
        CHECK_EQ(inputs.size(), 1);
        // just copy tensor from cpu to device
        context_->CopyCPUTensorToDevice(tensor_, outputs[0]);
    }

    void ShapeKernel::InferOutputShape(TensorShapeList& input_shapes,
            TensorShapeList& output_shapes){
        // set output dformat first, then we can according
        // to dformat to infer output shape
        output_tensor_dformats_.emplace_back(dlxnet::TensorProto::ANY4);

        output_shapes.clear();
        output_shapes.resize(1);
        CHECK_EQ(input_shapes.size(), 1);
        const int dims_size = input_shapes[0].size();
        output_shapes[0] = {dims_size};

        // tensor own data
        tensor_ = new Tensor(cpu_allocator(), Tensor::DT_FLOAT, output_shapes[0],
                dlxnet::TensorProto::ANY);
        for(int i=0;i<dims_size;++i){
            tensor_->host<float>()[i] = input_shapes[0][i];
        }
    }

    ShapeKernel::~ShapeKernel(){
        delete tensor_;
    }

    REGISTER_KERNEL_WITH_NAME(ShapeKernel, "Shape");
}//namespace opengl
