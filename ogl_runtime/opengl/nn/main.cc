#include <stdio.h>
#include <memory>
#include <iostream>
#include <random>
#include <assert.h>
#include <cmath>
#include <cstring>
#include "opengl/core/init.h"
#include "opengl/core/types.h"
#include "opengl/utils/util.h"
#include <glog/logging.h>

#include "opengl/core/fbo_session.h"
#include "opengl/utils/env_time.h"
#include <opencv2/opencv.hpp>

// only tensor is visible in nn module
using opengl::Tensor;
using opengl::TensorList;
using opengl::FBOSession;

Tensor* PrepareInputs(std::string image_fname,
        const std::vector<int>& shape){
    CHECK_GT(shape[0], 0);
    CHECK_GT(shape[1], 0);
    auto raw_image = cv::imread(image_fname);
    cv::cvtColor(raw_image, raw_image, CV_BGR2RGB);
    cv::resize(raw_image, raw_image, cv::Size(shape[0], shape[1]));
    raw_image.convertTo(raw_image, CV_32FC3);
    const float mean_vals[3] = { 123.f, 117.f, 104.f};
    raw_image = raw_image - cv::Scalar(mean_vals[0], mean_vals[1], mean_vals[2]);

    ::opengl::IntList input_shape({1, shape[1], shape[0], 3});
    ::opengl::DataFormat input_dformat = ::dlxnet::TensorProto::NHWC;
    Tensor* input_tensor= Tensor::Zeros(Tensor::DT_FLOAT,
            input_shape, input_dformat);
    ::memcpy(input_tensor->host(), raw_image.data,
            input_tensor->num_elements()*sizeof(float));

    return input_tensor;
}

int main(int argc, char** argv){
    // Initialize Google's logging library.
    google::InitGoogleLogging(argv[0]);

    // init opengl
    ::opengl::glfw_init();

    // init params
    std::string model_path = std::string(ROOT_DIR)+"/"+"./demo.dlx";
    const int num_iters = 1;
    int height=320;
    int width=320;
    std::string image_name = std::string(ROOT_DIR)+"/"+"../assets/container.jpg";
    if(argc>=2){
        model_path = std::string(argv[1]);
    }

    if(argc>=3){
        width = atoi(argv[2]);
        height = width;
    }

    // prepare inputs and outputs
    ::opengl::TensorList outputs_cpu;
    ::opengl::TensorNameList output_names({"cls_logits"});

    Tensor* input_tensor = PrepareInputs(image_name, {width, height});
    ::opengl::StringList dformats({"ANY"});

    auto session = std::unique_ptr<FBOSession>(new FBOSession);
    session->LoadGraph(model_path);

    // warming up
    for(int i=0;i<3;++i){
        // init graph according to inputs
        // and then do computation for the graph
        session->Run({{"input", input_tensor}});

        // get cpu outputs from device
        session->GetOutputs(output_names, dformats, &outputs_cpu);

    }

    auto env_time = EnvTime::Default();
    auto start_time1 = env_time->NowMicros();
    // std::string output_fn1 = "output1.txt";
    // std::string output_fn2 = "output2.txt";

    for(int i=0;i<num_iters;++i){
        // init graph according to inputs
        // do computation for the graph
        session->Run({{"input", input_tensor}});

        {
            // get cpu outputs from device
            session->GetOutputs(output_names, dformats, &outputs_cpu);
        }

        // print output
        LOG(INFO)<<outputs_cpu[0]->ShortDebugString();
        // dump output tensor
        // opengl::DumpTensor(outputs_cpu[0], output_fn1);
        // opengl::CompareTXT(output_fn1, output_fn2);
    }
    auto duration_time = env_time->NowMicros()-start_time1;
    // std::cout<<"Total Time: "<<duration_time*1e-3<<" ms\n";
    auto second_per_round = duration_time*1e-6/num_iters;
    // force to display
    std::cout<<"FPS: "<<1.0/second_per_round<<std::endl;

    LOG(INFO)<<"BiasAdd Success";

    return 0;
}
