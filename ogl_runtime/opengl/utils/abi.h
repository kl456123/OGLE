#ifndef OPENGL_UTILS_ABI_H_
#define OPENGL_UTILS_ABI_H_
#include "opengl/core/types.h"

namespace opengl{
    namespace port {

        string MaybeAbiDemangle(const char* name);

    }  // namespace port
} // namespace opengl


#endif
