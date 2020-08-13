#include "opengl/nn/kernels/conv2d.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"



namespace opengl{
    Conv2DKernel::Conv2DKernel(Context* context)
        :Kernel(context){
            // set work size
            for(int i=0;i<3;i++){
                work_sizes_[i] = 1;
            }
        }

    void Conv2DKernel::SelectKernel(const TensorList& inputs){
        if(inputs[0]->dformat()==dlxnet::TensorProto::ANY4){
            // as for different conv2d use different shaders
            if(kernel_size_==1&& group_size_==1&&dilation_==1&&stride_==1&&padding_==0){
                // 1x1 conv2d
                kernel_fname_=glsl_conv2d_pw_any4_glsl;
            }else if(group_size_!=1){
                // add insanity check here to make sure it is dw conv
                // consider it as depthwise conv2d
                CHECK_EQ(group_size_, inputs[0]->shape()[3]);
                kernel_fname_=glsl_conv2d_dw_any4_glsl;
            }else{
                // default conv2d shader
                kernel_fname_ = glsl_conv2d_any4_glsl;
            }
        }else{
            // as for different conv2d use different shaders
            if(kernel_size_==1&& group_size_==1&&dilation_==1&&stride_==1&&padding_==0){
                // 1x1 conv2d
                kernel_fname_=glsl_conv2d_pw_glsl;
            }else if(group_size_!=1){
                // add insanity check here to make sure it is dw conv
                // consider it as depthwise conv2d
                if(group_size_==inputs[0]->shape()[3]&&group_size_==inputs[1]->shape()[0]){
                    kernel_fname_=glsl_conv2d_dw_glsl;
                }else{
                    kernel_fname_ = glsl_conv2d_grp_glsl;
                }
            }else{
                // default conv2d shader
                kernel_fname_ = glsl_conv2d_glsl;
            }
        }

        // set build options here
        if(input_tensor_indexes_.size()==3){
            build_options_ += "#define USE_BIAS\n";
        }

        if(activation_type_=="Clip"){
            build_options_ += "#define USE_CLIP\n";
        }else if(activation_type_=="Relu"){
            build_options_ += "#define USE_RELU\n";
        }

        output_tensor_dformats_.emplace_back(inputs[0]->dformat());
    }

    void Conv2DKernel::SetupAttr(const dlxnet::Attribute& attr){
        auto& conv2d_params = attr.conv2d_attr();

        // handle pads
        CHECK_EQ(conv2d_params.pads_size(), 4);
        for(auto& pad: conv2d_params.pads()){
            CHECK_EQ(conv2d_params.pads(0),pad);
        }
        padding_ = conv2d_params.pads(0);

        // handle stride
        CHECK_EQ(conv2d_params.strides_size(), 2);
        CHECK_EQ(conv2d_params.strides(0), conv2d_params.strides(1));
        stride_ = conv2d_params.strides(0);


        // handle kernel
        CHECK_EQ(conv2d_params.kernel_shape_size(), 2);
        CHECK_EQ(conv2d_params.kernel_shape(0), conv2d_params.kernel_shape(1));
        kernel_size_=conv2d_params.kernel_shape(0);

        // set default for dilation and groups now
        group_size_=conv2d_params.group();
        if(group_size_==0){
            group_size_=1;
        }

        // CHECK_GT(group_size_, 0);
        if(conv2d_params.dilations_size()== 2){
            CHECK_EQ(conv2d_params.dilations(0), conv2d_params.dilations(1));
            dilation_=conv2d_params.dilations(0);
        }else{
            if(conv2d_params.dilations_size()== 1){
                dilation_ = conv2d_params.dilations(0);
            }else{
                // set default
                dilation_ = 1;
            }
        }

        // only used for fused op
        activation_type_= conv2d_params.activation_type();
        min_=conv2d_params.min();
        max_=conv2d_params.max();
    }


    Conv2DKernel::~Conv2DKernel(){}

    void Conv2DKernel::Compute(TensorList& inputs, TensorList& outputs){
        program_->Activate();
        auto input_image = inputs[0]->device<Texture>();
        auto input_filter = inputs[1]->device<Texture>();
        bool use_bias = inputs.size()>2;
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
        program_->set_int("group", group_size_);
        program_->set_int("dilation", dilation_);
        program_->set_int("use_bias", int(use_bias));
        if(!activation_type_.empty()){
            if(activation_type_=="Clip"){
                // clip case
                program_->set_int("act", 1);
            }else{
                program_->set_int("act", 2);
            }
            program_->set_float("min_value", min_);
            program_->set_float("max_value", max_);
        }else{
            program_->set_int("act", 0);
        }
        // input
        {
            program_->set_image2D("input_image", input_image->id(),  0);
            OPENGL_CHECK_ERROR;
        }

        // filter
        {
            program_->set_image2D("input_filter", input_filter->id(),  1);
            OPENGL_CHECK_ERROR;
        }
        if(use_bias){
            // bias
            auto input_bias = inputs[2]->device<Texture>();
            program_->set_image2D("input_bias", input_bias->id(),  2);
        }

        program_->Run();
    }

    void Conv2DKernel::InferOutputShape(TensorShapeList& input_shapes,
            TensorShapeList& output_shapes){
        // its order list as input, weights, bias
        CHECK(input_shapes.size()==3||input_shapes.size()==2);
        output_shapes.clear();
        output_shapes.resize(1);
        for(auto& input_shape:input_shapes){
            // check input is the same shape
        }
        // image_shape: (n, h, w, c)
        auto& image_shape = input_shapes[0];
        auto& filter_shape = input_shapes[1];
        CHECK_EQ(image_shape.size(), 4);
        CHECK_EQ(filter_shape.size(), 4);
        // check the conv2d parameters accordind to filter shapes
        // check filter is valid
        // filter_shape: (n_out, n_in , h, w)
        CHECK_EQ(filter_shape[2], kernel_size_);
        CHECK_EQ(filter_shape[3], kernel_size_);
        // channel should be the same with input image
        CHECK_EQ(filter_shape[1]*group_size_, image_shape[3]);

        const int kernel_size = kernel_size_ *dilation_-dilation_+1;
        const int output_height = (image_shape[1]-kernel_size+2*padding_)/stride_+1;
        const int output_width = (image_shape[2]-kernel_size+2*padding_)/stride_+1;
        output_shapes[0] = {image_shape[0], output_height, output_width, filter_shape[0]};
    }

    REGISTER_KERNEL_WITH_NAME(Conv2DKernel, "Conv");
}//namespace opengl



