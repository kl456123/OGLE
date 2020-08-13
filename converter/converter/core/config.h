#ifndef CONVERTER_CORE_CONFIG_H_
#define CONVERTER_CORE_CONFIG_H_
#include <string>

struct ConverterConfig{
    // where is the source model from
    enum class MODEL_SOURCE{ONNX=0, TENSORFLOW=1, CAFFE=2};
    ConverterConfig():src_model_path(),dst_model_path(){
    }

    // some configs
    std::string src_model_path;
    std::string dst_model_path;
    MODEL_SOURCE src;
};


#endif
