#ifndef OPENGL_UTILS_STRINGS_H_
#define OPENGL_UTILS_STRINGS_H_
#include <sstream>

#include "opengl/core/types.h"


namespace opengl{
    namespace strings{
        template<typename T>
            string StrCat(const T& arg){
                std::stringstream ss;
                ss<<arg;
                return ss.str();
            }

        template <typename T, typename ...Args>
            string StrCat(const T& arg1, const Args&... args){
                std::stringstream ss;
                ss<<arg1<<StrCat(args...);
                return ss.str();
            }

        void split(const std::string& s, std::vector<string>& tokens,
                const std::string& delimiters= " ");
    }// namespace strings
} // namespace opengl


#endif
