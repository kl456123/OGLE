#include <fstream>
#include <stdio.h>

#include "core/utils.h"

namespace utils{
    bool ONNXReadProtoFromBinary(const char* file_path, google::protobuf::Message* message){
        std::ifstream fs(file_path, std::ifstream::in | std::ifstream::binary);
        if (!fs.is_open()) {
            fprintf(stderr, "open failed %s\n", file_path);
            return false;
        }

        google::protobuf::io::IstreamInputStream input(&fs);
        google::protobuf::io::CodedInputStream codedstr(&input);

        codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);

        bool success = message->ParseFromCodedStream(&codedstr);

        fs.close();

        return success;
    }
}
