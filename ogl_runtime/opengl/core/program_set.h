#ifndef OPENGL_CORE_PROGRAM_SET_H_
#define OPENGL_CORE_PROGRAM_SET_H_
#include <map>

#include "opengl/core/types.h"

namespace opengl{
    class Program;
    class ProgramSet{
        public:
            ProgramSet();
        private:
            std::map<string, Program*> named_programs_;

    };
}//namespace opengl


#endif
