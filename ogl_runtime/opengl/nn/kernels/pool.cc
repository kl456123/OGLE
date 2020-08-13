#include "opengl/nn/kernels/pool.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"


namespace opengl{
    template<PoolType pool_type>
        PoolKernel<pool_type>::PoolKernel(Context* context)
        :Kernel(context){
        }

    template<PoolType pool_type>
        void PoolKernel<pool_type>::SelectKernel(const TensorList& inputs){
            if(inputs[0]->dformat()==dlxnet::TensorProto::ANY4){
                kernel_fname_ = "../opengl/nn/glsl/pool_any4.glsl";
            }else{
                kernel_fname_ = "../opengl/nn/glsl/pool.glsl";
            }
            // set single output dformat for all typed pool kernels
            output_tensor_dformats_.emplace_back(inputs[0]->dformat());
        }

    template<PoolType pool_type>
        void PoolKernel<pool_type>::SetupAttr(const dlxnet::Attribute& attr){

            if(pool_type_==GlobalAveragePool){
                return;
            }
            auto& maxpool_params = attr.maxpool_attr();

            // handle pads
            CHECK_EQ(maxpool_params.pads_size(), 4);
            for(auto& pad: maxpool_params.pads()){
                CHECK_EQ(maxpool_params.pads(0),pad);
            }
            padding_ = maxpool_params.pads(0);

            // handle stride
            CHECK_EQ(maxpool_params.strides_size(), 2);
            CHECK_EQ(maxpool_params.strides(0), maxpool_params.strides(1));
            stride_ = maxpool_params.strides(0);


            // handle kernel
            CHECK_EQ(maxpool_params.kernel_shape_size(), 2);
            CHECK_EQ(maxpool_params.kernel_shape(0), maxpool_params.kernel_shape(1));
            kernel_size_=maxpool_params.kernel_shape(0);
        }
    template<PoolType pool_type>
        void PoolKernel<pool_type>::Compute(TensorList& inputs, TensorList& outputs){
            program_->Activate();
            auto input_image = inputs[0]->device<Texture>();
            program_->SetRetVal(outputs);

            auto input_shape = inputs[0]->shape();
            auto output_shape = outputs[0]->shape();

            program_->set_vec3i("input_shape", input_shape[1],
                    input_shape[2], input_shape[3]);
            program_->set_vec3i("output_shape", output_shape[1],
                    output_shape[2], output_shape[3]);
            program_->set_int("padding", padding_);
            program_->set_int("kernel_size", kernel_size_);
            program_->set_int("stride_size", stride_);
            program_->set_int("pool_type", pool_type_);
            // input
            {
                program_->set_image2D("input_image", input_image->id(),  0);
                OPENGL_CHECK_ERROR;
            }

            program_->Run();
        }
    template<PoolType pool_type>
        void PoolKernel<pool_type>::InferOutputShape(TensorShapeList& input_shapes,
                TensorShapeList& output_shapes){
            CHECK_EQ(input_shapes.size(), 1);
            output_shapes.clear();
            output_shapes.resize(1);
            auto& image_shape = input_shapes[0];
            if(pool_type_==GlobalAveragePool){
                output_shapes[0]={image_shape[0],1, 1, image_shape[3]};
                // set pool params according to the input shape
                stride_=1;
                // spatial dims
                CHECK_EQ(image_shape[1], image_shape[2]);
                kernel_size_=image_shape[1];
                padding_=0;
            }else{
                // compute output shape like conv2d
                const int output_height = (image_shape[1]-kernel_size_+2*padding_)/stride_+1;
                const int output_width = (image_shape[2]-kernel_size_+2*padding_)/stride_+1;
                output_shapes[0] = {image_shape[0], output_height, output_width, image_shape[3]};
            }

        }

    template<PoolType pool_type>
        void PoolKernel<pool_type>::InferOutputShape(const TensorList& input_tensors,
                TensorShapeList& output_shapes){
            CHECK_EQ(input_tensors.size(), 1);
            output_shapes.clear();
            output_shapes.resize(1);
            if(pool_type_==GlobalAveragePool){
                output_shapes[0]={input_tensors[0]->shape()[0], 1, 1, input_tensors[0]->shape()[3]};
                // set pool params according to the input shape
                stride_=1;
                // spatial dims
                CHECK_EQ(input_tensors[0]->shape()[2], input_tensors[0]->shape()[1]);
                kernel_size_=input_tensors[0]->shape()[2];
                padding_=0;
            }else{
                // compute output shape like conv2d
                const int output_height = (input_tensors[0]->shape()[1]-kernel_size_+2*padding_)/stride_+1;
                const int output_width = (input_tensors[0]->shape()[2]-kernel_size_+2*padding_)/stride_+1;
                output_shapes[0] = {input_tensors[0]->shape()[0], output_height,
                    output_width, input_tensors[0]->shape()[3]};
            }
        }

    template<PoolType pool_type>
        PoolKernel<pool_type>::~PoolKernel(){}

    REGISTER_KERNEL_WITH_NAME(PoolKernel<MaxPool>, "MaxPool");
    REGISTER_KERNEL_WITH_NAME(PoolKernel<AveragePool>, "AveragePool");
    REGISTER_KERNEL_WITH_NAME(PoolKernel<GlobalAveragePool>, "GlobalAveragePool");
}//namespace opengl
