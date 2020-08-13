#include <fstream>
#include <fstream>

#include <fcntl.h>
#include <glog/logging.h>
#include "opengl/utils/protobuf.h"


namespace opengl{
    bool ReadProtoFromBinary(const char* file_path, google::protobuf::Message* message){
        std::ifstream fs(file_path, std::ifstream::in | std::ifstream::binary);
        if (!fs.is_open()) {
            LOG(FATAL)<<"open failed "<< file_path;
            return false;
        }

        google::protobuf::io::IstreamInputStream input(&fs);
        google::protobuf::io::CodedInputStream codedstr(&input);

        codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);

        bool success = message->ParseFromCodedStream(&codedstr);

        fs.close();

        return success;
    }

    void ReadProtoFromText(const std::string& fn, google::protobuf::Message* message){
        int fd = open(fn.c_str(), O_RDONLY);
        CHECK(fd>0)<<"error when opening demo.cfg.";
        google::protobuf::io::FileInputStream input(fd);
        input.SetCloseOnDelete(true);
        google::protobuf::TextFormat::Parse(&input, message);
    }

    void WriteProtoToText(const std::string& fn, google::protobuf::Message& message){
        std::string str_proto;
        google::protobuf::TextFormat::PrintToString(message, &str_proto);
        std::fstream output(fn, std::ios::out|std::ios_base::ate);
        CHECK(output)<<"error during saving to txt.";
        output<<str_proto;
        output.flush();
        output.close();
    }

    void WriteProtoToBinary(const std::string& fn, google::protobuf::Message& message){
        std::fstream output(fn, std::ios::out | std::ios::trunc | std::ios::binary);
        CHECK(output)<<"error during saving to binary.";
        message.SerializeToOstream(&output);
        output.flush();
        output.close();
    }
}//namespace opengl
