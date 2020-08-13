#include <glog/logging.h>

#include "opengl/core/buffer.h"
#include "opengl/utils/macros.h"

namespace opengl{
    Buffer::Buffer(GLsizeiptr size, GLenum target, GLenum usage, float* data){
        target_ = target;
        size_ = size;

        // create new name of buffer object
        glGenBuffers(1, &id_);
        // assign to context
        glBindBuffer(target, id_);
        // allocate mem in device
        glBufferData(target, size_, data, usage);
    }

    Buffer::~Buffer(){
        glDeleteBuffers(1, &id_);
    }


    void* Buffer::Map(GLbitfield bufMask){
        glBindBuffer(target_, id_);
        OPENGL_CHECK_ERROR;
        // (TODO figure out 0 meaning)
        auto ptr = glMapBufferRange(target_, 0, size_, bufMask);
        OPENGL_CHECK_ERROR;
        return ptr;
    }


    void Buffer::UnMap(){
        glBindBuffer(target_, id_);
        OPENGL_CHECK_ERROR;
        glUnmapBuffer(target_);
        OPENGL_CHECK_ERROR;
    }
}//namespace opengl
