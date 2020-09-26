#include "opengl/nn/kernels/const.h"
#include "opengl/core/kernel_registry.h"
#include "opengl/core/context.h"
#include "opengl/core/kernel.h"

namespace opengl{
    void ConstKernel::SetupAttr(const dlxnet::Attribute& attr){
        // get value and store it in tensor_
        auto& const_tensor = attr.const_attr().value();

        CHECK(!tensor_);
        tensor_ = new Tensor(const_tensor);
        CHECK_GE(tensor_->shape().size(), 0)<<DebugString();

        output_tensor_dformats_.emplace_back(const_tensor.target_data_format());
    }

    ConstKernel::ConstKernel(Context* context)
        :Kernel(context),tensor_(nullptr){}

    void ConstKernel::Compute(TensorList& inputs, TensorList& outputs){
        // set value from tensor_
        // for now just copy host memory data to device
        CHECK(inputs.size()==0)<<"Input should be empty in ConstKernel";
        DLOG(INFO)<<"Load Weights From CPU in Const Kernel: "<<kernel_name_;

        context_->CopyCPUTensorToDevice(tensor_, outputs[0]);
    }

    void ConstKernel::InferOutputShape(TensorShapeList& inputs,
            TensorShapeList& outputs){
        // check input tensors is empty
        CHECK(inputs.size()==0)<<"Input should be empty in ConstKernel";
        CHECK_EQ(output_tensor_dformats_.size(), 1);
        auto dformat = output_tensor_dformats_[0];

        std::vector<int> output_shape;
        // check it is image shape first(dims size is 4)
        if(dformat==dlxnet::TensorProto::ANY4){
            output_shape=tensor_->shape();
        }else{
            // change shape according to target dformat
            if(dformat==dlxnet::TensorProto::NHWC4){
                LOG(FATAL)<<"Removed dformat: "
                    << dformat << " in Const Kernel";
                // nhwc
                // output_shape = {tensor_->num(), tensor_->height(),
                    // tensor_->width(), tensor_->channel()};
            }else if(dformat==dlxnet::TensorProto::HWN4C4){
                output_shape = tensor_->shape();
                // nchw
                // output_shape = {tensor_->num(), tensor_->channel(),
                    // tensor_->height(), tensor_->width()};
            }else{
                LOG(FATAL)<<"unsupported dformat "
                    << dformat << " in Const Kernel";
            }
        }
        // get shape from outputs
        outputs.emplace_back(output_shape);
    }

    ConstKernel::~ConstKernel(){
        delete tensor_;
    }

    REGISTER_KERNEL_WITH_NAME(ConstKernel, "Const");
    REGISTER_KERNEL_WITH_NAME(ConstKernel, "Constant");
}//namespace opengl
