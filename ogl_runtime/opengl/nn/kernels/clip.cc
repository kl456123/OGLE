#include "opengl/nn/kernels/clip.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"


namespace opengl{
    ClipKernel::ClipKernel(Context* context)
        :Kernel(context){
            kernel_fname_ = "../opengl/nn/glsl/clip.glsl";
        }

    void ClipKernel::SetupAttr(const dlxnet::Attribute& attr){
        auto& clip_params = attr.clip_attr();
        min_ = clip_params.min();
        max_ = clip_params.max();
    }

    void ClipKernel::Compute(TensorList& inputs, TensorList& outputs){
        DLOG(INFO)<<"ClipKernel Inputs: "<<inputs.size();
        program_->Activate();
        auto input_image = inputs[0]->device<Texture>();

        program_->SetRetVal(outputs);
        program_->set_float("min_value", min_);
        program_->set_float("max_value", max_);

        // input
        {
            program_->set_image2D("input_image", input_image->id(),  0);
            OPENGL_CHECK_ERROR;
        }

        program_->Run();
    }

    void ClipKernel::InferOutputShape(const TensorList& inputs,
            TensorShapeList& output_shapes){
        CHECK_EQ(inputs.size(), 1);
        // set dformat
        output_tensor_dformats_.emplace_back(inputs[0]->dformat());

        // set shape
        output_shapes.clear();
        output_shapes.resize(1);
        output_shapes[0] = inputs[0]->shape();
    }

    ClipKernel::~ClipKernel(){}

    REGISTER_KERNEL_WITH_NAME(ClipKernel, "Clip");
}//namespace opengl
