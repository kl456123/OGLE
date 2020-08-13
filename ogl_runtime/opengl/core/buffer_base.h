#ifndef BUFFER_BASE_H_
#define BUFFER_BASE_H_
#include "opengl.h"

namespace opengl{
    class BufferBase{
        public:
            BufferBase(GLuint id, GLenum target);
            virtual ~BufferBase(){};
            // accessor
            GLuint id()const{return id_;}
            GLenum target()const{return target_;}
        protected:
            GLuint id_;
            GLenum target_;
    };
}//namespace opengl

#endif
