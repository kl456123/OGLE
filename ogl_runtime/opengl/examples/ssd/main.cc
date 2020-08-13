#include "opengl/utils/logging.h"
#include <string.h>
#include "opengl/core/fbo_session.h"
#include "opengl/core/tensor.h"
#include "opengl/core/init.h"
#include "opengl/core/functor.h"
#include "opengl/utils/util.h"
#include "opengl/examples/ssd/detector.h"

#include <opencv2/opencv.hpp>
#include "opengl/examples/ssd/common.h"

using opengl::FBOSession;
using opengl::Tensor;
using opengl::Detector;

namespace functor = opengl::functor;

int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);
    ::opengl::glfw_init();

    // config detector
    std::string model_path = opengl::GenerateAbsPath("./demo.dlx");
    auto detector = Detector::Create(model_path, {"input"},
            {"cls_logits", "box_preds", "anchors"}, {160, 160});

    auto cap = std::shared_ptr<cv::VideoCapture>(reinterpret_cast<cv::VideoCapture*>(open_video_stream("", -1, 640, 480, 0)));
    std::vector<std::string> image_name_list;
    cv::Mat raw_image;
    std::string images_dir = "/home/indemind/images";

    //ReadFilesFromDir(images_dir, &image_name_list);
    // while(true){
    // for(auto& image_name: image_name_list){
    // auto image_path = images_dir+"/"+image_name;
    // raw_image = cv::imread(image_path);
    // if(raw_image.data==nullptr){
    // LOG(WARNING)<<"wrong image format or image not found: "<<image_path;
    // continue;
    // }

    // auto t1 = std::chrono::system_clock::now();
    // // detect
    // std::vector<BoxInfo> finalBoxInfos;

    // detector->Detect(raw_image, finalBoxInfos);
    // auto t2 = std::chrono::system_clock::now();
    // float dur = (float)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000;
    // std::cout << "duration time:" << dur << "ms" << std::endl;

    // drawBoxes(finalBoxInfos, raw_image);
    // cv::namedWindow("dlxnet", CV_WINDOW_NORMAL);
    // cv::imshow("dlxnet", raw_image);
    // cv::waitKey(0);
    // }
    // }

    while(true){
        // prepare inputs
        // std::string image_fname = "../opengl/examples/ssd/000000145679.jpg";
        // raw_image = cv::imread(image_fname);
        *(cap.get()) >> raw_image;

        auto t1 = std::chrono::system_clock::now();
        // detect
        std::vector<BoxInfo> finalBoxInfos;

        detector->Detect(raw_image, finalBoxInfos);
        auto t2 = std::chrono::system_clock::now();
        float dur = (float)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000;
        std::cout << "duration time:" << dur << "ms" << std::endl;

        drawBoxes(finalBoxInfos, raw_image);
        cv::namedWindow("dlxnet", CV_WINDOW_NORMAL);
        cv::imshow("dlxnet", raw_image);
        cv::waitKey(1);
    }
    return 0;
}
