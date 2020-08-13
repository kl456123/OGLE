#include "opengl/core/program.h"
#include "opengl/core/init.h"
#include "opengl/test/test.h"
#include "opengl/utils/macros.h"

using namespace opengl::testing;


namespace opengl{
    namespace{
        const std::string kSourceFname = "../opengl/examples/gpgpu/buffer2image.glsl";

        auto initializer = [](){
            // init window and context
            auto window = glfw_init(1280, 800);

            // init glew
            glew_init();
        };
    }// namespace


    TEST(Program, AttachFile){
        initializer();
        auto program = Program();
        EXPECT_FALSE(program.program_id()==0);
        // program.AttachFile(kSourceFname);
        // EXPECT_OPENGL_NO_ERROR;
        // program.Link();
        // EXPECT_OPENGL_NO_ERROR;
        // program.Activate();
        // EXPECT_OPENGL_NO_ERROR;
    }

    TEST(Program, AttachMultipleFiles){
    }

    TEST(Program, SetArgs){
    }


}// namespace opengl

