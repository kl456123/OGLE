#include "opengl/nn/kernels/decoder.h"

#include "opengl/core/program.h"
#include "opengl/core/fbo_session.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"
#include "opengl/utils/util.h"


namespace opengl{
    DecoderKernel::DecoderKernel(Context* context)
        :Kernel(context){
            // set predefined parameters for common ssd detector
            score_threshold_=0.3f;
            nms_threshold_=0.45f;
            variances_={0.1, 0.1, 0.2, 0.2};
        }

    void DecoderKernel::SelectKernel(const TensorList& inputs){
        CHECK_EQ(inputs.size(), 2);
        CHECK_EQ(inputs[0]->dformat(), dlxnet::TensorProto::ANY4);
        kernel_fname_ = "../opengl/nn/glsl/decoder.glsl";

        output_tensor_dformats_.emplace_back(dlxnet::TensorProto::ANY4);
    }

    void DecoderKernel::SetupAttr(const dlxnet::Attribute& attr){
        // no attr used here
    }

    void DecoderKernel::Compute(TensorList& inputs, TensorList& outputs){
        DLOG(INFO)<<"DecoderKernel Inputs: "<<inputs.size();
        program_->Activate();
        auto prediction = inputs[0]->device<Texture>();
        auto anchors = inputs[1]->device<Texture>();
        program_->SetRetVal(outputs);


        auto input_shape = inputs[0]->shape();
        auto output_shape = outputs[0]->shape();

        program_->set_vec4("variances", variances_);
        program_->set_vec4i("input_shape", input_shape);

        // input
        {
            program_->set_image2D("prediction", prediction->id(),  0);
            OPENGL_CHECK_ERROR;
        }

        // filter
        {
            program_->set_image2D("anchors", anchors->id(),  1);
            OPENGL_CHECK_ERROR;
        }

        program_->Run();
    }




    void DecoderKernel::InferOutputShape(const TensorList& input_tensors,
            TensorShapeList& output_shapes){
        CHECK_GE(input_tensors.size(), 2);
        auto prediction = input_tensors[0];
        auto anchors = input_tensors[1];

        const int num_batches = prediction->shape()[0];
        const int num_samples = prediction->shape()[1];

        // how many features to represent single sample
        // num_classes + 4
        const int num_entries = prediction->last_stride();

        // assume anchor is (N, H, W, 4)

        auto anchors_shape = anchors->shape();
        // TODO(breakpoint change default shape for anchor
        // to make it as (N, NUM_SAMPLES, 4)
        CHECK_EQ(anchors_shape.size(), 4);
        CHECK_EQ(anchors_shape[0], num_batches);
        CHECK_EQ(anchors_shape[1] * anchors_shape[2], num_samples);
        CHECK_EQ(anchors_shape[3], 4);

        output_shapes.clear();
        output_shapes.resize(1);

        // (N, FINAL_NUM_SAMPLES, 4)
        output_shapes[0] = {num_batches, num_samples, 4};
    }

    DecoderKernel::~DecoderKernel(){}

    REGISTER_KERNEL_WITH_NAME(DecoderKernel, "Decoder");
} // namespace opengl
