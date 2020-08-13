#ifndef OPENGL_CORE_INIT_H_
#define OPENGL_CORE_INIT_H_
#include "opengl/core/opengl.h"


namespace opengl{
    // init some context ,windows and loaded function
    int glew_init();
    void log_glinfo();

    GLFWwindow* glfw_init(const int width=1280, const int height=800);

    // init some buffer objects(XBO) and texture objects(XTO)

    GLuint InitPBO();

    GLuint InitSSBO(int size);

}//namespace opengl
#endif
