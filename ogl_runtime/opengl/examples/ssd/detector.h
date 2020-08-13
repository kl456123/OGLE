#ifndef OPENGL_EXAMPLES_SSD_DETECTOR_H_
#define OPENGL_EXAMPLES_SSD_DETECTOR_H_
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include "opengl/examples/ssd/common.h"
#include "opengl/core/types.h"

#include <chrono>


namespace opengl{
    struct DetectorOptions{
        string model_name;
        int input_width;
        int input_height;
        float nms_threshold;
        float score_threshold;
        int topk;
        std::vector<float> variances;
        TensorNameList input_names;
        TensorNameList output_names;
    };
    class FBOSession;

    class Detector{
        public:
            virtual void Preprocess(const cv::Mat& image_in, cv::Mat& image_out);

            void NMS(std::vector<BoxInfo>& boxInfos,std::vector<BoxInfo>& boxInfos_left, float threshold);
            virtual void Detect(const cv::Mat& raw_image, std::vector<BoxInfo>& finalBoxInfos);


            virtual ~Detector();

            static std::unique_ptr<Detector> Create(const string& model_name,
                    const TensorNameList& input_names, const TensorNameList& output_names,
                    const IntList& shape);
            static std::unique_ptr<Detector> Create(const DetectorOptions& options);

            void LoadToOutputTensors(const TensorNameList& names);
            void PrintDetectorInfo();


        private:
            explicit Detector(const DetectorOptions& options);
            // only uesd in postprocess
            void GetTopK(std::vector<BoxInfo>& input, int top_k);
            void GenerateBoxInfo(std::vector<BoxInfo>& boxInfos, float score_threshold);
            void LoadToInputTensors(const cv::Mat& raw_image);
            void Run(const cv::Mat& raw_image);

            std::vector<float> variances_;
            std::string model_name_;

            int topk_;
            float score_threshold_;
            float nms_threshold_;

            // num of total classes including bg
            int num_classes_;

            // hw
            IntList input_sizes_;
            IntList origin_input_sizes_;

            // input and output tensors and its names
            std::vector<::opengl::Tensor*> output_tensors_;
            std::vector<::opengl::Tensor*> input_tensors_;
            std::vector<std::string> input_names_;
            std::vector<std::string> output_names_;

            std::unique_ptr<FBOSession> session_;

            // store intermedia tmp results
            std::vector<::opengl::Tensor*> tmp_tensors_;
    };

}//namespace opengl

#endif
