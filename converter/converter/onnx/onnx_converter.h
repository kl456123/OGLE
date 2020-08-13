#ifndef CONVERTER_ONNX_CONVERTER_H_
#define CONVERTER_ONNX_CONVERTER_H_
#include "core/converter.h"
#include "onnx.pb.h"



class ONNXConverter: public Converter{
    public:
        ONNXConverter()=default;
        virtual ~ONNXConverter(){}
        virtual void Run()override;
        virtual void Reset(const ConverterConfig config);

        void PrintSelf(){}
};


REGISTER_CLASS_CONVERTER(ONNXConverter);

#endif
