#ifndef CONVERTER_CORE_UTILS_H_
#define CONVERTER_CORE_UTILS_H_
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

namespace utils{
    bool ONNXReadProtoFromBinary(const char* file_path, google::protobuf::Message* message);
}// namespace utils


#endif
