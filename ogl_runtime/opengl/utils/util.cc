#include <sstream>
#include <fstream>
#include <iomanip>
#include <experimental/filesystem>

#include "opengl/core/opengl.h"
#include "opengl/utils/util.h"
#include "opengl/utils/logging.h"
#include "opengl/core/tensor.h"


namespace opengl{
    namespace fs = std::experimental::filesystem;

    void setLocalSize(std::vector<std::string>& prefix, int* localSize,
            std::vector<int> local_sizes){
        GLint maxLocalSizeX, maxLocalSizeY, maxLocalSizeZ;
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &maxLocalSizeX);
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &maxLocalSizeY);
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &maxLocalSizeZ);

        localSize[0]     = local_sizes[0] < maxLocalSizeX ? local_sizes[0] : maxLocalSizeX;
        localSize[1]     = local_sizes[1] < maxLocalSizeY ? local_sizes[1] : maxLocalSizeY;
        localSize[2]     = local_sizes[2] < maxLocalSizeZ ? local_sizes[2] : maxLocalSizeZ;

        {
            std::ostringstream os;
            os << "#define XLOCAL " << localSize[0];
            prefix.push_back(os.str());
        }
        {
            std::ostringstream os;
            os << "#define YLOCAL " << localSize[1];
            prefix.push_back(os.str());
        }
        {
            std::ostringstream os;
            os << "#define ZLOCAL " << localSize[2];
            prefix.push_back(os.str());
        }

    }

    IntList AmendShape(const IntList& shape, const int amend_size){
        // CHECK_LE(shape.size(), amend_size);
        IntList amended_shape;
        if(amend_size<shape.size()){
            for(int i=shape.size()-amend_size;i<shape.size();++i){
                amended_shape.emplace_back(shape[i]);
            }
            return amended_shape;
        }
        const int remain_dims = amend_size-shape.size();
        amended_shape = shape;
        for(int i=0;i<remain_dims;++i){
            amended_shape.insert(amended_shape.begin(), 1);
        }
        return amended_shape;
    }


    void DumpTensor(const Tensor* tensor, const string& output_fn){
        std::ofstream file;
        file.open(output_fn, std::ofstream::out);
        if(file.fail()){
            LOG(FATAL)<<"Open file Error: "<<output_fn;
        }
        const float* data = tensor->host<float>();
        const int num_elements = tensor->num_elements();
        auto shape = tensor->shape();
        const int dims_size = shape.size();
        for(int i=0;i<num_elements;++i){
            file<<std::fixed<<std::setprecision(8)<<data[i]<<"\n";
        }
        file.close();
    }


    void CompareTXT(const string& output_fn1, const string& output_fn2,
            const float precision){
        std::ifstream file1;
        file1.open(output_fn1, std::ifstream::in);
        if(file1.fail()){
            LOG(FATAL)<<"Open file Error: "<<output_fn1;
        }

        std::ifstream file2;
        file2.open(output_fn2, std::ifstream::in);
        if(file2.fail()){
            LOG(FATAL)<<"Open file Error: "<<output_fn2;
        }
        string s1, s2;
        float val1, val2;
        int index = 0;
        while(file1&&file2){
            file1>>val1;
            file2>>val2;
            CHECK_LE((val1), (val2)+(precision))<<" in index: "<<index;
            CHECK_GE((val1), (val2)-(precision))<<" in index: "<<index;
            ++index;
        }
    }


    string GenerateAbsPath(const string& fname){
        fs::path p(fname);
        return fs::absolute(p);
    }
}//namespace opengl

