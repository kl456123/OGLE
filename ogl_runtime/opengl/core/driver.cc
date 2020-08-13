#include <string.h>
#include <glog/logging.h>
#include <memory>

#include "opengl/core/driver.h"

namespace opengl{
    GLuint CreateTexture2D(GLsizei width, GLsizei height,
            const void *data, GLenum format,
            GLenum dtype, GLenum internal_format){
        GLuint texture;

        // Create a texture.
        OPENGL_CALL(glGenTextures(1, &texture));

        VLOG(1) << "Created 2D texture [" << texture << "]";

        // Bind to temporary unit.
        glBindTexture(GL_TEXTURE_2D, texture);

        // Similar to cudaMemcpy.
        OPENGL_CALL(glTexImage2D(GL_TEXTURE_2D, /*level=*/0, internal_format,
                    width, height, /*border=*/0,
                    format, dtype, nullptr));
        OPENGL_CALL(
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
        OPENGL_CALL(
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
        OPENGL_CALL(
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
        OPENGL_CALL(
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));

        if(data){
            OPENGL_CALL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                        format, dtype, data));
        }

        return texture;
    }

    GLuint CreateTexture3D(GLsizei width, GLsizei height,
            GLsizei depth, const void *data, GLenum format,
            GLenum dtype, GLenum internal_format){
        GLuint texture;
        // Create a texture.
        OPENGL_CALL(glGenTextures(1, &texture));
        VLOG(1) << "Created 3D texture [" << texture << "]";
        glBindTexture(GL_TEXTURE_3D, texture);

        OPENGL_CALL(glTexImage3D(GL_TEXTURE_3D, /*level=*/0, internal_format,
                    width, height, depth, /*border=*/0,
                    format, dtype, nullptr));

        OPENGL_CALL(
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
        OPENGL_CALL(
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
        OPENGL_CALL(
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE));
        OPENGL_CALL(
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
        OPENGL_CALL(
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));

        if(data){
            OPENGL_CALL(glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0,0, width, height, depth,
                        format, dtype, data));
        }
        return texture;
    }

    GLuint CreateFrameBuffer(){
        GLuint frame_buffer;
        OPENGL_CALL(glGenFramebuffers(1, &frame_buffer));

        // Create frame buffer And Check its Completation
        OPENGL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer));
        // Set the list of draw buffers.
        GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
        // "1" is the size of DrawBuffers.
        OPENGL_CALL(glDrawBuffers(1, DrawBuffers));
        return frame_buffer;
    }

    GLuint CreatePixelBuffer(size_t bytes, bool read){
        // create buffer first
        GLuint io_buffer;
        OPENGL_CALL(glGenBuffers(1, &io_buffer));
        // set property according to readable or writeable
        if(read){
            OPENGL_CALL(glBindBuffer(GL_PIXEL_PACK_BUFFER, io_buffer));
            OPENGL_CALL(glBufferData(GL_PIXEL_PACK_BUFFER, bytes,
                        NULL, GL_STREAM_READ));
        }else{
            OPENGL_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, io_buffer));
            OPENGL_CALL(glBufferData(GL_PIXEL_UNPACK_BUFFER, bytes,
                        NULL, GL_STREAM_DRAW));
        }
        return io_buffer;
    }


    int GetNumOfChannels(GLenum format){
        switch(format){
            case GL_RGBA:
                return 4;
            case GL_RED:
                return 1;
            default:
                LOG(FATAL)<<"Unsupported format: "<<format;
                return -1;
        }
    }

    void CopyHostToTexture(const void* data, GLint width, GLint height, GLuint texture,
            GLenum format, GLenum dtype){
        // default bind texture to texture_2d
        OPENGL_CALL(glBindTexture(GL_TEXTURE_2D, texture));
        OPENGL_CALL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    width, height, format, dtype, data));
    }

    void CopyTextureToHost(void* data, GLint width, GLint height, GLuint texture,
            GLenum format, GLenum dtype){
        GLint ext_format, ext_type;
        ext_format = format;
        ext_type = dtype;
        //TODO(breakpoint) fix the bug here
        // OPENGL_CALL(glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &ext_format));
        // OPENGL_CALL(glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &ext_type));
        OPENGL_CALL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                    texture, 0));
        CHECK_EQ(ext_type, dtype)<<"unmatched type";
        CHECK_EQ(ext_format, format)<<"unmatched format";
        OPENGL_CALL(glReadBuffer(GL_COLOR_ATTACHMENT0));
        OPENGL_CALL(glReadPixels(0, 0, width, height, ext_format, ext_type, data));
    }

    void DeleteBuffer(GLuint buffer){
        OPENGL_CALL(glDeleteBuffers(1, &buffer));
    }

    void DeleteFrameBuffer(GLuint frame_buffer){
        OPENGL_CALL(glDeleteFramebuffers(1, &frame_buffer));
    }


    void CopyHostToTextureDMA(const void* data, GLint width, GLint height, GLuint texture,
            GLenum format, GLenum dtype){
        GLuint io_buffer;
        OPENGL_CALL(glGenBuffers(1, &io_buffer));
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, io_buffer);
        size_t bytes = width * height * GetNumOfChannels(format) * sizeof(float);
        OPENGL_CALL(glBufferData(GL_PIXEL_UNPACK_BUFFER, bytes, NULL, GL_STREAM_DRAW));

        // copy data from host to pbo
        void* mem = glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, bytes, GL_MAP_WRITE_BIT);
        CHECK_NOTNULL(mem);
        memcpy(mem, data, bytes);
        OPENGL_CALL(glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER));
        // copy pbo to texture
        OPENGL_CALL(glBindTexture(GL_TEXTURE_2D, texture));
        OPENGL_CALL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    width, height, format, dtype, 0));
        OPENGL_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
    }

    void CopyTextureToHostDMA(void* data, GLint width, GLint height, GLuint texture,
            GLenum format, GLenum dtype){
        GLint ext_format, ext_type;
        OPENGL_CALL(glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &ext_format));
        OPENGL_CALL(glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &ext_type));
        CHECK_EQ(ext_type, dtype)<<"unmatched type";
        CHECK_EQ(ext_format, format)<<"unmatched format";

        size_t bytes = width*height*GetNumOfChannels(format)*sizeof(float);
        GLuint io_buffer;
        OPENGL_CALL(glGenBuffers(1, &io_buffer));

        // specify which texture to read
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                texture , 0);

        OPENGL_CALL(glReadBuffer(GL_COLOR_ATTACHMENT0));
        OPENGL_CALL(glBindBuffer(GL_PIXEL_PACK_BUFFER, io_buffer));
        OPENGL_CALL(glBufferData(GL_PIXEL_PACK_BUFFER, bytes,
                    NULL, GL_STREAM_READ));
        // copy data from fbo to pbo
        OPENGL_CALL(glReadPixels(0, 0, width, height, ext_format, ext_type,
                    0));
        // copy data from pbo to host
        void* mem = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, bytes, GL_MAP_READ_BIT);
        CHECK_NOTNULL(mem);
        memcpy(data, mem, bytes);
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    }
    int GetMaxTextureSize(){
        int work_grp_inv;
        OPENGL_CALL(glGetIntegerv(GL_MAX_TEXTURE_SIZE, &work_grp_inv));
        return work_grp_inv;
    }

    void AttachTextureToFrameBuffer(GLuint texture, GLint width, GLint height){
        OPENGL_CALL(glViewport(0, 0, width, height));
        // Set "renderedTexture" as our colour attachement #0
        OPENGL_CALL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                    texture , 0));

        // Always check that our framebuffer is ok
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            LOG(FATAL) << "Framebuffer not complete.";
        }
    }

    GLuint CreateShader(GLenum shader_kind, const char *shader_src) {
        // Create the shader.
        GLuint shader = glCreateShader(shader_kind);
        OPENGL_CALL(glShaderSource(shader, 1, &shader_src, nullptr));
        OPENGL_CALL(glCompileShader(shader));

        // Check compile errors.
        GLint err;
        OPENGL_CALL(glGetShaderiv(shader, GL_COMPILE_STATUS, &err));

        GLint info_log_len;
        OPENGL_CALL(glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_len));

        if (info_log_len > 0) {
            std::unique_ptr<char[]> err_msg(new char[info_log_len + 1]);
            OPENGL_CALL(glGetShaderInfoLog(shader, info_log_len, nullptr, err_msg.get()));
            LOG(FATAL) << err_msg.get();
        }

        return shader;
    }


    void Sync(){
        OPENGL_CALL(glFinish());
    }

    void LogSystemInfo(){
        printf("OpenGL info:\n"
                "\tVendor   = \"%s\"\n"
                "\tRenderer = \"%s\"\n"
                "\tVersion  = \"%s\"\n"
                "\tGLSL     = \"%s\"\n",
                glGetString(GL_VENDOR),
                glGetString(GL_RENDERER),
                glGetString(GL_VERSION),
                glGetString(GL_SHADING_LANGUAGE_VERSION)
              );
    }
}//namespace opengl
