#include "opengl/nn/kernels/gather.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"


namespace opengl{
    GatherKernel::GatherKernel(Context* context)
        :Kernel(context){
            kernel_fname_ = glsl_gather_glsl;
        }

    void GatherKernel::SetupAttr(const dlxnet::Attribute& attr){
        auto& gather_params = attr.gather_attr();
        axis_= gather_params.axis();
    }

    void GatherKernel::Compute(TensorList& inputs, TensorList& outputs){
        DLOG(INFO)<<"GatherKernel Inputs: "<<inputs.size();
        program_->Activate();
        auto input_image = inputs[0]->device<Texture>();
        auto input_index = inputs[1]->device<Texture>();

        program_->SetRetVal(outputs);
        program_->set_int("axis", axis_);

        // input
        {
            program_->set_image2D("input_image", input_image->id(),  0);
            OPENGL_CHECK_ERROR;
        }

        {
            program_->set_image2D("input_index", input_index->id(),  1);
            OPENGL_CHECK_ERROR;
        }

        program_->Run();
    }

    void GatherKernel::InferOutputShape(TensorShapeList& input_shapes,
            TensorShapeList& output_shapes){
        // set output dformat first, then we can according
        // to dformat to infer output shape
        output_tensor_dformats_.emplace_back(dlxnet::TensorProto::ANY4);

        output_shapes.clear();
        output_shapes.resize(1);
        CHECK_EQ(input_shapes.size(), 2);
        // output shape equals to input index
        // only single number is supported
        CHECK_EQ(input_shapes[1].size(), 1);
        CHECK_EQ(input_shapes[1][0], 1);
        output_shapes[0] = input_shapes[1];
    }

    GatherKernel::~GatherKernel(){}

    REGISTER_KERNEL_WITH_NAME(GatherKernel, "Gather");
}//namespace opengl
