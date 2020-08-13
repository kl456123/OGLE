#include "opengl/nn/kernels/batchnorm.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"


namespace opengl{
    BatchNormKernel::BatchNormKernel(Context* context)
        :Kernel(context){
        }

    void BatchNormKernel::SetupAttr(const dlxnet::Attribute& attr){
        auto& bn_params = attr.batchnorm_attr();
        momentum_ = bn_params.momentum();
        eps_ = bn_params.epsilon();


    }
    void BatchNormKernel::SelectKernel(const TensorList& inputs){
        if(inputs[0]->dformat()==dlxnet::TensorProto::ANY4){
            kernel_fname_ = "../opengl/nn/glsl/batchnorm_any4.glsl";
        }else{
            kernel_fname_ = "../opengl/nn/glsl/batchnorm.glsl";
        }
        // single output
        output_tensor_dformats_.emplace_back(inputs[0]->dformat());
    }

    void BatchNormKernel::Compute(TensorList& inputs, TensorList& outputs){
        DLOG(INFO)<<"BatchNormalization Inputs: "<<inputs.size();
        program_->Activate();
        auto input_image = inputs[0]->device<Texture>();
        auto gamma = inputs[1]->device<Texture>();
        auto beta = inputs[2]->device<Texture>();
        auto mean = inputs[3]->device<Texture>();
        auto var = inputs[4]->device<Texture>();

        program_->SetRetVal(outputs);
        program_->set_float("eps", eps_);
        program_->set_float("momentum", momentum_);
        program_->set_vec3i("output_shape", outputs[0]->height(),
                outputs[0]->width(), outputs[0]->channel());

        // input
        {
            program_->set_image2D("input_image", input_image->id(),  0);
            OPENGL_CHECK_ERROR;
        }

        // gamma
        {
            program_->set_image2D("input_gamma", gamma->id(), 1);
            OPENGL_CHECK_ERROR;
        }

        // beta
        {
            program_->set_image2D("input_beta", beta->id(), 2);
            OPENGL_CHECK_ERROR;
        }

        // mean
        {
            program_->set_image2D("input_mean", mean->id(), 3);
            OPENGL_CHECK_ERROR;
        }

        // variance
        {
            program_->set_image2D("input_var", var->id(), 4);
            OPENGL_CHECK_ERROR;
        }

        program_->Run();
    }

    void BatchNormKernel::InferOutputShape(TensorShapeList& input_shapes,
            TensorShapeList& output_shapes){
        CHECK_EQ(input_shapes.size(), 5);
        output_shapes.clear();
        output_shapes.emplace_back(input_shapes[0]);
    }

    void BatchNormKernel::InferOutputShape(const TensorList& input_tensors,
            TensorShapeList& output_shapes){
        CHECK_EQ(input_tensors.size(), 5);
        output_shapes.clear();
        output_shapes.emplace_back(input_tensors[0]->shape());
    }

    BatchNormKernel::~BatchNormKernel(){}

    REGISTER_KERNEL_WITH_NAME(BatchNormKernel, "BatchNormalization");
}//namespace opengl
