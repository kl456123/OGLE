/* Image Classification Task, Input a single image,  Output confidences for multiple
 * predefined classes(1000 classes in ImageNet).
 *
 */
#include <glog/logging.h>
#include <string.h>
#include "opengl/core/fbo_session.h"
#include "opengl/core/tensor.h"
#include "opengl/core/init.h"

#include <opencv2/opencv.hpp>

using opengl::FBOSession;
using opengl::Tensor;

bool ReadImage(const char* image_name, Tensor** out){
    // read image and store them in image ptr in nhwc dformat
    cv::Mat raw_image = cv::imread(image_name);
    if(raw_image.data==nullptr){
        return false;
    }
    // preprocess image using mean and std
    cv::Mat image;
    cv::cvtColor(raw_image,image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(224, 224));
    image.convertTo(image, CV_32FC3);

    const float mean_vals[3] = { 0.485, 0.456, 0.406};
    const float std_vals[3] = {0.229, 0.224, 0.225};
    auto mean = cv::Scalar(mean_vals[0], mean_vals[1], mean_vals[2]);
    auto stds = cv::Scalar(std_vals[0], std_vals[1], std_vals[2]);
    // use cv::divide to prevent from template deduction problem
    // in arm platform
    image = (1/255.0 * image - mean);
    cv::divide(image, stds, image);


    std::vector<int>shape = {1, 224, 224, 3};
    // copy to tensor
    Tensor* tensor = new Tensor(Tensor::DT_FLOAT, shape);
    // Tensor* tensor = Tensor::Ones(Tensor::DT_FLOAT, shape);
    ::memcpy(tensor->host(), image.data, tensor->RequestedSize());

    // assign to out
    *out = tensor;
    return true;
}


int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);
    ::opengl::glfw_init();

    // load model
    std::string model_path = "./demo.dlx";
    auto session = std::unique_ptr<FBOSession>(new FBOSession);
    session->LoadGraph(model_path);

    // prepare input
    // load image to input tensor
    const char* image_name= "../opengl/examples/label_image/cat.jpg";
    Tensor* image_ptr=nullptr;
    bool success = ReadImage(image_name, &image_ptr);
    if(!success){
        LOG(FATAL)<<"Read Image Failed";
        return -1;
    }

    ::opengl::TensorList outputs_cpu;

    // Run Session with input
    // do computation for the graph
    session->Run({{"input", image_ptr}});

    // get cpu outputs from device
    session->GetOutputs({"output"}, {"NHWC"}, &outputs_cpu);

    // print output
    // LOG(INFO)<<outputs_cpu[0]->ShortDebugString();
    // find the max index
    const int num_elements = outputs_cpu[0]->num_elements();
    const float* data = outputs_cpu[0]->host<float>();
    float max_value = data[0];
    int max_index = 0;
    for(int i=0;i<num_elements;++i){
        if(max_value<data[i]){
            max_value = data[i];
            max_index = i;
        }
    }
    LOG(INFO)<<"Max Confidence Index: "<<max_index;
    return 0;
}
