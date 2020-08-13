#include "opengl/utils/macros.h"


namespace opengl{
    const char *GLGetErrorString(GLenum error) {
        switch (error) {
            case GL_NO_ERROR:
                return "GL_NO_ERROR";
            case GL_INVALID_ENUM:
                return "GL_INVALID_ENUM";
            case GL_INVALID_VALUE:
                return "GL_INVALID_VALUE";
            case GL_INVALID_OPERATION:
                return "GL_INVALID_OPERATION";
            case GL_OUT_OF_MEMORY:
                return "GL_OUT_OF_MEMORY";
            default:
                return "Unknown OpenGL error code";
        }
    }
    void OpenGLCheckErrorWithLocation(const char* fname, int line){
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            LOG(FATAL) << "OpenGL error, code=" << err << ": "
                << GLGetErrorString(err)<<"\nin "<<fname
                <<" : "<<line << std::endl;
        }
    }
}//namespace opengl
