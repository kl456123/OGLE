#include "opengl/core/buffer.h"
#include "opengl/core/init.h"
#include "opengl/test/test.h"
#include "opengl/utils/macros.h"


using namespace opengl::testing;


namespace opengl{
    namespace{
        // opengl init
        auto res = [](){
            glfw_init();
            glew_init();
            return 0;
        }();


        // buffer read and write test
        const int num = 1<<3;
        const int size = num*sizeof(float);
        auto buffer_device = ShaderBuffer(size);
        float buffer_cpu1[num]={0};
        float buffer_cpu2[num]={0};

        void clear_buffer(){
            memset(buffer_cpu1, 0, sizeof(buffer_cpu1));
            memset(buffer_cpu2, 0, sizeof(buffer_cpu2));
        }

        void init_buffer(){
            for(int i=0;i<num;i++){
                buffer_cpu1[i] = random()%100;
                buffer_cpu2[i] = random()%100;
            }
        }

        void upload(){
            //write first from cpu1
            ::memcpy(buffer_device.Map(GL_MAP_WRITE_BIT), buffer_cpu1, size);
            buffer_device.UnMap();
        }
        void download(){
            // read then to cpu2
            ::memcpy(buffer_cpu2, buffer_device.Map(GL_MAP_WRITE_BIT), size);
            buffer_device.UnMap();
        }
    }// namespace

    TEST(Buffer, Upload){
        upload();
        EXPECT_OPENGL_NO_ERROR;
    }

    TEST(Buffer, Download){
        download();
        EXPECT_OPENGL_NO_ERROR;
    }

    TEST(Buffer, Correctness){
        init_buffer();
        upload();
        download();
        // check the same between cpu1 and cpu2
        for(int i=0;i<num;++i){
            EXPECT_EQ(buffer_cpu1[i], buffer_cpu2[i]);
        }
    }
}//namespace opengl
