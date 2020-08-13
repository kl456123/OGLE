#ifndef OPENGL_CORE_SCOPE_H_
#define OPENGL_CORE_SCOPE_H_
#include <vector>
#include <string>
#include "opengl/core/dlxnet.pb.h"

namespace opengl{
    // construct graph manually using api
    class Scope{
        public:
            Scope();

            ~Scope();


            ::dlxnet::NodeProto* AddNode();

            void AddInputName(const std::string& name);

            void AddOutputName(const std::string& name);

            int AddTensor(const std::string& tensor_name);

            const dlxnet::ModelProto ExportModel()const{
                return *model_;
            }

            // accessor
            const ::dlxnet::GraphProto& graph()const{
                return model_->graph();
            }
        private:
            int tensor_index_;
            dlxnet::ModelProto* model_;
    };
}


#endif
