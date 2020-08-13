#ifndef UTIL_H_
#define UTIL_H_
#include <string>
#include <vector>
#include "opengl/core/types.h"


namespace opengl{
    void setLocalSize(std::vector<std::string>& prefix, int* localSize, std::vector<int> local_sizes);
    IntList AmendShape(const IntList& shape, const int amend_size=4);

    void DumpTensor(const Tensor* tensor, const string& output_fn);

    void CompareTXT(const string& output_fn1, const string& output_fn2,
            const float precision=1e-3);
    string GenerateAbsPath(const string& fname);
}//namespace opengl

#endif
