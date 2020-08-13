#include "opengl/nn/kernels/binary.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"



namespace opengl{
    BinaryKernel::BinaryKernel(Context* context)
        :Kernel(context){
            // set work size
            for(int i=0;i<3;i++){
                work_sizes_[i] = 1;
            }
        }


    BinaryKernel::~BinaryKernel(){}
    void BinaryKernel::SelectKernel(const TensorList& inputs){
        kernel_fname_ = glsl_binary_glsl;
        output_tensor_dformats_.emplace_back(inputs[0]->dformat());
    }

    void BinaryKernel::Compute(TensorList& inputs, TensorList& outputs){
        program_->Activate();
        auto input0 = inputs[0]->device<Texture>();
        auto input1 = inputs[1]->device<Texture>();
        program_->SetRetVal(outputs);


        auto input_shape = inputs[0]->shape();
        auto output_shape = outputs[0]->shape();

        program_->set_vec3i("input_shape", input_shape[1], input_shape[2], input_shape[3]);
        // input
        {
            program_->set_image2D("input0", input0->id(),  0);
            OPENGL_CHECK_ERROR;
        }

        // filter
        {
            program_->set_image2D("input1", input1->id(),  1);
            OPENGL_CHECK_ERROR;
        }

        program_->Run();
    }

    void BinaryKernel::InferOutputShape(const TensorList& input_tensors,
            TensorShapeList& output_shapes){
        output_shapes.clear();
        output_shapes.emplace_back(input_tensors[0]->shape());
    }

    void BinaryKernel::InferOutputShape(TensorShapeList& input_shapes,
            TensorShapeList& output_shapes){
        output_shapes.clear();
        output_shapes.resize(1);
        for(auto& input_shape:input_shapes){
            // check input is the same shape
        }
        output_shapes[0] = input_shapes[0];
    }

    void BinaryKernel::SetupAttr(const dlxnet::Attribute& attr){
    }

    REGISTER_KERNEL_WITH_NAME(BinaryKernel, "Add");
}//namespace opengl



