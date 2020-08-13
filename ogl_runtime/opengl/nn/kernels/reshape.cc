#include "opengl/nn/kernels/reshape.h"
#include "opengl/core/fbo_session.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"
#include "opengl/utils/util.h"


namespace opengl{
    ReshapeKernel::ReshapeKernel(Context* context)
        :Kernel(context){
        }

    void ReshapeKernel::SelectKernel(const TensorList& inputs){
        if(inputs[0]->dformat()==dlxnet::TensorProto::ANY4){
            kernel_fname_ = glsl_reshape_any4_glsl;
        }else{
            CHECK_EQ(inputs[0]->dformat(), dlxnet::TensorProto::NHWC4);
            kernel_fname_ = glsl_reshape_glsl;
        }
        // always output any4 dformat
        output_tensor_dformats_.emplace_back(dlxnet::TensorProto::ANY4);
    }

    void ReshapeKernel::SetupAttr(const dlxnet::Attribute& attr){
    }

    void ReshapeKernel::Compute(TensorList& inputs, TensorList& outputs){
        DLOG(INFO)<<"ReshapeKernel Inputs: "<<inputs.size();
        program_->Activate();
        auto input_image = inputs[0]->device<Texture>();
        auto shape_image = inputs[1]->device<Texture>();

        program_->SetRetVal(outputs);
        program_->set_vec4i("input_shape", AmendShape(inputs[0]->shape()));
        program_->set_vec4i("output_shape", AmendShape(target_shape_));

        // input
        {
            program_->set_image2D("input_image", input_image->id(),  0);
            OPENGL_CHECK_ERROR;
        }

        program_->Run();
    }

    void ReshapeKernel::InferOutputShape(const TensorList& inputs,
            TensorShapeList& output_shapes){
        output_shapes.clear();
        output_shapes.resize(1);
        CHECK_EQ(inputs.size(), 2);
        // make sure it is prepared(computed)
        auto shape_tensor = inputs[1];
        auto& output_shape = output_shapes[0];

        Tensor* cpu_tensor = new Tensor(Tensor::DT_FLOAT, shape_tensor->shape(),
                Tensor::HOST_MEMORY, dlxnet::TensorProto::ANY);
        session_->context()->CopyDeviceTensorToCPU(shape_tensor, cpu_tensor);
        for(int i=0;i<cpu_tensor->num_elements();++i){
            output_shape.emplace_back(cpu_tensor->host<float>()[i]);
        }

        // preprocess some dims that has special meanings like 0 and -1
        auto input_shape = inputs[0]->shape();
        auto num_elements = inputs[0]->num_elements();
        int remain_elements = 1;
        for(int i=0;i<output_shape.size();++i){
            if(output_shape[i]==0){
                output_shape[i] = input_shape[i];
            }
            if(output_shape[i]!=-1){
                remain_elements*=output_shape[i];
            }
        }
        bool flag = true;
        for(int i=0;i<output_shape.size();++i){
            if(output_shape[i]==-1){
                CHECK(flag);
                output_shape[i] = num_elements/remain_elements;
                flag=false;
            }
        }

        CHECK_EQ(target_shape_.size(), 0);
        target_shape_ = output_shape;

        delete cpu_tensor;
    }


    ReshapeKernel::~ReshapeKernel(){}

    REGISTER_KERNEL_WITH_NAME(ReshapeKernel, "Reshape");
}//namespace opengl

