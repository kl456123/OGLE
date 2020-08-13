#include "opengl/core/session.h"
#include "opengl/nn/kernels/binary.h"
#include "opengl/core/kernel_registry.h"


namespace opengl{
    Session::Session(Context* context)
        :context_(context){}

    void Session::LoadGraph(KernelList kernels){
        kernels_ = kernels;
    }

    void Session::LoadGraph(StringList kernel_names){
        // init kernels first
        kernels_.clear();
        kernels_.reserve(kernel_names.size());

        // create each kernel
        Kernel* kernel=nullptr;
        for(auto& kernel_name:kernel_names){
            KernelRegistry::Global()->CreateKernel(kernel_name, &kernel, context_);
            if(kernel==nullptr){
                LOG(FATAL)<<"unsupported kernel name "<<kernel_name;
            }
            // if(kernel_name=="add"){
            // kernel = new BinaryKernel(context_);
            // }
            kernels_.emplace_back(kernel);
        }
    }

    Session::Session(){
        context_ = new Context(nullptr);
    }

    void Session::Run(){
        for(auto& kernel: kernels_){
            kernel->Compute(texture_inputs_, texture_outputs_);
        }
    }

    void Session::Setup(TensorList inputs_cpu){
        // upload to gpu
        texture_inputs_.clear();
        texture_inputs_.resize(inputs_cpu.size());

        // setup input
        for(unsigned int i=0;i<inputs_cpu.size();++i){
            CHECK(inputs_cpu[i]->is_host());
            auto texture_input = new Tensor(Tensor::DT_FLOAT, inputs_cpu[i]->shape(),
                    Tensor::DEVICE_TEXTURE);
            context_->CopyCPUTensorToDevice(inputs_cpu[i], texture_input);
            texture_inputs_[i] = texture_input;
        }

        auto output_shape = inputs_cpu[0]->shape();
        TensorShapeList input_shapes, output_shapes;
        for(auto& input_cpu:inputs_cpu){
            input_shapes.emplace_back(input_cpu->shape());
        }

        // use the last layer
        Kernel* kernel = kernels_[kernels_.size()-1];

        // get output_shape
        kernel->InferOutputShape(input_shapes,
                output_shapes);

        // allocate output texture according to their shape
        texture_outputs_.clear();
        texture_outputs_.reserve(output_shapes.size());
        for(auto& output_shape: output_shapes){
            texture_outputs_.emplace_back(new Tensor(Tensor::DT_FLOAT, output_shape,
                        Tensor::DEVICE_TEXTURE));
        }
    }

    void Session::GetOutputs(TensorList outputs){
        CHECK(outputs.size()==texture_outputs_.size())
            <<"outputs size is not equal to outputs from session";
        for(unsigned int i=0;i<texture_outputs_.size();++i){
            context_->CopyDeviceTensorToCPU(texture_outputs_[i], outputs[i]);
        }
    }
}//namespace opengl
