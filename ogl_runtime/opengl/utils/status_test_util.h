#ifndef OPENGL_UTILS_STATUS_TEST_UTIL_H_
#define OPENGL_UTILS_STATUS_TEST_UTIL_H_

#include "opengl/utils/status.h"
#include "opengl/test/test.h"

// Macros for testing the results of functions that return tensorflow::Status.
#define DLXNET_EXPECT_OK(statement) \
    EXPECT_EQ(::opengl::Status::OK(), (statement))
#define DLXNET_ASSERT_OK(statement) \
    ASSERT_EQ(::opengl::Status::OK(), (statement))


#endif
