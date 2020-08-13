#ifndef OEPNGL_UTILS_PROTOBUF_H_
#define OEPNGL_UTILS_PROTOBUF_H_
/**
 * This file contains Utility functions about protobuf.
 * like Decode and Encode protos
 *
 */
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

namespace opengl{
    // read functions
    bool ReadProtoFromBinary(const char* file_path, google::protobuf::Message* message);
    void ReadProtoFromText(const std::string& fn, google::protobuf::Message* message);

    // write functions
    void WriteProtoToText(const std::string& fn, google::protobuf::Message& message);
    void WriteProtoToBinary(const std::string& fn, google::protobuf::Message& message);
}//namespace opengl


#endif

