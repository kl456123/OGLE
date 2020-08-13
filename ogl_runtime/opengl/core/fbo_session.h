#ifndef OPENGL_CORE_FBO_SESSION_H_
#define OPENGL_CORE_FBO_SESSION_H_

#include "opengl/core/types.h"
#include "opengl/core/context.h"
#include "opengl/core/dlxnet.pb.h"

namespace opengl{
    class Env;

    class FBOSession{
        public:
            FBOSession(Context* context);
            FBOSession():FBOSession(GetContext()){}
            virtual ~FBOSession();

            /*! use inputs to allocate tensor, prepare all memory
             * to run late
             */
            void Setup(const NamedTensorList& inputs_cpu);

            /*!
             * Draw texture to framebuffer, then
             */
            void Run(const NamedTensorList& inputs_cpu);

            // load graph from literal in memory
            void LoadGraph(const ::dlxnet::ModelProto& model_proto);

            void LoadGraph(const ::dlxnet::ModelProto&& model_proto);

            // load graph from protobuf binary in disk
            void LoadGraph(std::string model_path);

            void GetOutputs(const TensorNameList& output_names,
                    const StringList& output_dformats, TensorList* outputs);

            std::string DebugString()const;
            Context* context()const{
                return context_;
            }

            bool IsONNX()const{
                return model_->producer_name()=="ONNX";
            }

            Tensor* FindTensorByName(const std::string& name);

            bool allow_resized(){return allow_resized_;}
            bool set_allow_resized(bool flag){allow_resized_=flag;}
            const IntList& default_input_sizes(){return default_input_sizes_;}

        private:

            // reorder all nodes in nodes_ topologically
            void TopologicalSort();

            bool CheckKernelReady(const Kernel* kernel);
            void UpdateKernelReady(const Kernel* kernel);

            // caller does not own it
            Tensor* FindTensorById(const int id);

            Context* context_;
            OwnedKernelList kernels_;

            dlxnet::ModelProto* model_;

            // contains all tensors used in the session
            // may be some slots are null due to that pruned and optimization
            OwnedTensorList total_tensors_;

            // check session is freezed or not
            // note that when graph is freezed, session can be called multiple times
            bool finalized_ = false;

            bool graph_created_= false;

            // map from tensor name to index in total_tensors_
            NamedIndex tensor_name_index_;

            std::vector<bool> ready_;

            friend class Kernel;
            Env* env_;

            // input shapes info
            bool allow_resized_=false;
            IntList default_input_sizes_;
    };

    void SetTrackingStats(bool flag);
}//namespace opengl


#endif
