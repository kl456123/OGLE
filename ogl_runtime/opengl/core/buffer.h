////////////////////////////////////////////////////
// Buffer is used to intermediate storage to upload
// or download
////////////////////////////////////////////////////

#ifndef BUFFER_H_
#define BUFFER_H_

#include "opengl/core/buffer_base.h"

namespace opengl{
    class Buffer{
        public:
            Buffer(GLsizeiptr size, GLenum type, GLenum usage, float* data);
            ~Buffer();

            // map to copy
            void* Map(GLbitfield bufMask);
            void UnMap();

            GLuint id()const{return id_;}
            GLsizeiptr size()const{return size_;}
            GLenum target()const{return target_;}
        private:
            GLsizeiptr size_;

            GLuint id_;
            GLenum target_;


    };

    class ShaderBuffer: public Buffer{
        public:
            ShaderBuffer(GLsizeiptr size, float* data=nullptr)
                :Buffer(size, GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, data){
                }
    };
}//namespace opengl

#endif
