#include "opengl/utils/strings.h"


namespace opengl{
    namespace strings{
        void split(const std::string& s, std::vector<string>& tokens,
                const std::string& delimiters){
            string::size_type lastpos = s.find_first_not_of(delimiters, 0);
            string::size_type pos = s.find_first_of(delimiters, lastpos);
            while(std::string::npos!=pos||std::string::npos!=lastpos){
                tokens.push_back(s.substr(lastpos, pos-lastpos));
                lastpos = s.find_first_not_of(delimiters, pos);
                pos = s.find_first_of(delimiters, lastpos);
            }
        }

    } // namespace strings
} // namespace opengl
