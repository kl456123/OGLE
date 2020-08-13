#include "opengl/core/texture.h"
#include "opengl/utils/macros.h"
#include "opengl/core/driver.h"

namespace opengl{
    namespace{
        const int kMaxChannelSize = 4;
    }
    Texture::Texture(std::vector<int> dims, GLenum internal_format,
            GLenum target, float* image_data)
        :dims_(dims){
            // just 3d texture
            target_ = target;

            // automatically activate the texture at the same time
            OPENGL_CALL(glGenTextures(1, &id_));
            OPENGL_CALL(glBindTexture(target, id_));
            // change internal field for the object
            OPENGL_CALL(glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
            OPENGL_CALL(glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
            // STR
            OPENGL_CALL(glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
            OPENGL_CALL(glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));



            GLenum format = GL_RGBA;

            type_ = GL_FLOAT;
            CHECK_EQ(dims.size(), 2);
            CHECK_LE(dims[0], GetMaxTextureSize());
            CHECK_LE(dims[1], GetMaxTextureSize());
            CHECK_GT(dims[0], 0);
            CHECK_GT(dims[1], 0);

            OPENGL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, internal_format, dims[0], dims[1],
                        0, format, type_, image_data));

            // store internal format
            format_ = format;
            bytes_ = dims[0]*dims[1]*kMaxChannelSize * sizeof(float);
        }

    Texture::~Texture(){
        OPENGL_CALL(glDeleteTextures(1, &id_));
    }
}//namespace opengl
