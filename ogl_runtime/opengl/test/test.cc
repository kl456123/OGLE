#include "opengl/test/test.h"
#include <fstream>

namespace opengl{
    namespace testing{
        bool ReadStrFromFile(std::string fname, std::string* content){
            std::ifstream fd(fname);
            *content = std::string(std::istreambuf_iterator<char>(fd),
                    (std::istreambuf_iterator<char>()));
            if(content->empty()){
                return false;
            }
            return true;
        }
    }
}
