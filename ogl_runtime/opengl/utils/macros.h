#ifndef OPENGL_UTILS_MACROS_H_
#define OPENGL_UTILS_MACROS_H_
#include <glog/logging.h>

#include "opengl/core/opengl.h"

namespace opengl{
    void OpenGLCheckErrorWithLocation(const char* fname, int line);
    const char *GLGetErrorString(GLenum error);
}//opengl

/*!
 * \brief Protected OpenGL call.
 * \param func Expression to call.
 */
#define OPENGL_CALL(func)                                                      \
    do{                                                                            \
        (func);                                                                    \
        ::opengl::OpenGLCheckErrorWithLocation(__FILE__, __LINE__);                  \
    }while(false)

#ifdef ENABLE_OPENGL_CHECK_ERROR
#define OPENGL_CHECK_ERROR              \
{                                   \
    GLenum error = glGetError();    \
    if (GL_NO_ERROR != error){       \
        LOG(FATAL)<<"error here"; \
    }\
}
#else
#define OPENGL_CHECK_ERROR
#endif

#define EXPECT_OPENGL_NO_ERROR  \
    EXPECT_TRUE(glGetError()==GL_NO_ERROR)

#define ASSERT_OPENGL_NO_ERROR  \
    ASSERT_TRUE(glGetError()==GL_NO_ERROR)


#define UP_DIV(x, y)   (((x) + (y) - (1)) / (y))
#define UP_ROUND(x, y) (((x) + (y) - (1)) / (y) * (y))

// A macro to disallow the copy constructor and operator= functions
// This is usually placed in the private: declarations for a class.
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName&) = delete;         \
    void operator=(const TypeName&) = delete

#ifdef __has_builtin
#define HAS_BUILTIN(x) __has_builtin(x)
#else
#define HAS_BUILTIN(x) 0
#endif

// Compilers can be told that a certain branch is not likely to be taken
// (for instance, a CHECK failure), and use that information in static
// analysis. Giving it this information can help it optimize for the
// common case in the absence of better information (ie.
// -fprofile-arcs).
//
// We need to disable this for GPU builds, though, since nvcc8 and older
// don't recognize `__builtin_expect` as a builtin, and fail compilation.
#if (!defined(__NVCC__)) && \
    (HAS_BUILTIN(__builtin_expect) || (defined(__GNUC__) && __GNUC__ >= 3))
#define PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#else
#define PREDICT_FALSE(x) (x)
#define PREDICT_TRUE(x) (x)
#endif


#define TO_STRING(ROOT_DIR) #ROOT_DIR

#endif
