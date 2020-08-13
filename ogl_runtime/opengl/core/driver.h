#ifndef OPENGL_CORE_DRIVER_H_
#define OPENGL_CORE_DRIVER_H_
#include "opengl/core/init.h"
#include "opengl/utils/macros.h"


namespace opengl{
    // create memory object in opengl
    //
    // create texture 3d or 2d
    GLuint CreateTexture2D(GLsizei width, GLsizei height,
            const void *data, GLenum format=GL_RGBA,
            GLenum dtype=GL_FLOAT, GLenum internal_format=GL_RGBA32F);
    GLuint CreateTexture3D(GLsizei width, GLsizei height,
            GLsizei depth, const void *data, GLenum format=GL_RGBA,
            GLenum dtype=GL_FLOAT, GLenum internal_format=GL_RGBA32F);

    int GetNumOfChannels(GLenum format);

    int GetMaxTextureSize();

    // create buffer object
    // there are many bo in opengl like
    // fbo(frame buffer object), pbo(pixel buffer object)
    // vbo(vertex buffer object) and so on.
    GLuint CreatePixelBuffer(size_t bytes, bool read);
    GLuint CreateFrameBuffer();

    // clean buffer
    void DeleteBuffer(GLuint buffer);
    void DeleteFrameBuffer(GLuint frame_buffer);

    void AttachTextureToFrameBuffer(GLuint texture, GLint width, GLint height);
    GLuint CreateShader(GLenum shader_kind, const char *shader_src);


    void CopyHostToTexture(const void* data, GLint width, GLint height, GLuint texture,
            GLenum format=GL_RGBA, GLenum dtype=GL_FLOAT);
    void CopyTextureToHost(void* data, GLint width, GLint height, GLuint texture,
            GLenum format=GL_RGBA, GLenum dtype=GL_FLOAT);


    // DMA version of copy data between cpu and device
    void CopyHostToTextureDMA(const void* data, GLint width, GLint height, GLuint texture,
            GLenum format=GL_RGBA, GLenum dtype=GL_FLOAT);
    void CopyTextureToHostDMA(void* data, GLint width, GLint height, GLuint texture,
            GLenum format=GL_RGBA, GLenum dtype=GL_FLOAT);

    void Sync();
    void LogSystemInfo();
}
#endif
