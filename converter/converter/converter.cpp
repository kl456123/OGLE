#include <iostream>
#include <glog/logging.h>

#include "core/config.h"
#include "core/optimizer.h"
#include "core/converter.h"


int main(int argc,char** argv){
    // get converter_config from parser
    // here we just assign it manualy
    if(argc<2){
        std::cerr<<"argument is too few"<<std::endl;
        return -1;
    }

    ConverterConfig converter_config;
    converter_config.src_model_path = argv[1];

    // here use dlx as suffix, dlx refers to dlx framework
    converter_config.dst_model_path = "demo.dlx";
    converter_config.src = ConverterConfig::MODEL_SOURCE::ONNX;

    auto converter_registry = Registry<Converter>::Global();
    std::string format_name = "ONNXConverter";
    Converter* converter=nullptr;
    converter_registry->LookUp(format_name, &converter);

    CHECK_NOTNULL(converter);

    // init converter
    converter->Reset(converter_config);
    converter->Run();

    // use custom optimizer
    // add optimizer config
    auto optimizer = Optimizer::Global();

    converter->Optimize(optimizer);

    // save model proto to binary file
    converter->Save();

    // LOG(INFO)<<converter->DebugString();

    return 0;
}
