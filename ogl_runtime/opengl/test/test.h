#ifndef OPENGL_TEST_TEST_H_
#define OPENGL_TEST_TEST_H_
#include <gtest/gtest.h>
#include <string>
#include "opengl/utils/status.h"

namespace opengl{
    namespace testing{
        // some test utils here
        bool ReadStrFromFile(std::string fname, std::string* content);
    }
}

#endif
