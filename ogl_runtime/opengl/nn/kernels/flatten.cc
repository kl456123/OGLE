#include "opengl/nn/kernels/flatten.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"
#include "opengl/utils/util.h"
#include "opengl/core/fbo_session.h"
#include "opengl/core/functor.h"


namespace opengl{
    FlattenKernel::FlattenKernel(Context* context)
        :Kernel(context){}

    void FlattenKernel::SelectKernel(const TensorList& inputs){
        kernel_fname_ = "../opengl/nn/glsl/reshape.glsl";
        if(session_->IsONNX()&&inputs[0]->dformat()==dlxnet::TensorProto::NHWC4){
            IntList mapping{0, 3, 1, 2};
            axis_ = mapping[axis_];
        }
    }

    void FlattenKernel::SetupAttr(const dlxnet::Attribute& attr){
        auto& flatten_params = attr.flatten_attr();
        axis_ = flatten_params.axis();
    }

    void FlattenKernel::Compute(TensorList& inputs, TensorList& outputs){
        DLOG(INFO)<<"FlattenKernel Inputs: "<<inputs.size();

        Tensor* any4_tensor = nullptr;
        auto input_tensor = inputs[0];
        if(input_tensor->dformat() == dlxnet::TensorProto::NHWC4){
            VLOG(1)<<"Convert Tensor From NHWC4 To ANY4";
            if(!any4_tensor_){
                // cache here
                any4_tensor = new Tensor(Tensor::DT_FLOAT, input_tensor->shape(),
                        Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY4);
            }else{
                any4_tensor = any4_tensor_;
            }
            functor::ConvertTensorNHWC4ToANY4()(GetContext(), input_tensor, any4_tensor);
        }else{
            any4_tensor = inputs[0];
        }
        CHECK_EQ(any4_tensor->dformat(), dlxnet::TensorProto::ANY4);

        program_->Activate();
        auto input_image = any4_tensor->device<Texture>();

        program_->SetRetVal(outputs);
        program_->set_vec4i("input_shape", AmendShape(any4_tensor->shape()));
        program_->set_vec4i("output_shape", AmendShape(outputs[0]->shape()));

        // input
        {
            program_->set_image2D("input_image", input_image->id(),  0);
            OPENGL_CHECK_ERROR;
        }

        program_->Run();
    }

    void FlattenKernel::InferOutputShape(TensorShapeList& input_shapes,
            TensorShapeList& output_shapes){
        // set output dformat first, then we can according
        // to dformat to infer output shape
        output_tensor_dformats_.emplace_back(dlxnet::TensorProto::ANY4);

        // single output
        CHECK_EQ(input_shapes.size(), 1);
        output_shapes.clear();
        output_shapes.resize(1);
        CHECK_EQ(input_shapes[0].size(), 4);
        CHECK_EQ(output_shapes[0].size(), 0);
        const int start_dim = axis_;

        int num_elements=1;
        for(int i=start_dim;i<input_shapes[0].size();++i){
            num_elements*=input_shapes[0][i];
        }
        for(int i=0;i<start_dim;++i){
            output_shapes[0].emplace_back(input_shapes[0][i]);
        }
        output_shapes[0].emplace_back(num_elements);
    }

    FlattenKernel::~FlattenKernel(){
        if(any4_tensor_){
            delete any4_tensor_;
        }
    }

    REGISTER_KERNEL_WITH_NAME(FlattenKernel, "Flatten");
}//namespace opengl
