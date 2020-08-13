#include <sstream>

#include "opengl/core/fbo_session.h"
#include "opengl/utils/macros.h"
#include "opengl/utils/env.h"
#include "opengl/core/kernel.h"
#include "opengl/core/kernel_registry.h"
#include "opengl/utils/protobuf.h"
#include "opengl/core/tensor.h"
#include "opengl/core/tensor_format.h"
#include "opengl/core/driver.h"
#include "opengl/core/metric.h"
#include "opengl/core/step_stats_collector.h"
#include "opengl/core/step_stats.pb.h"
#include "opengl/utils/env.h"
#include "opengl/nn/profiler/traceme.h"


namespace opengl{
    namespace nodestats{
        inline int64 NowInNsec() { return EnvTime::Default()->NowNanos(); }

        void SetScheduled(NodeExecStatsInterface* stats, int64 micros) {
            if (!stats) return;
            stats->SetScheduled(micros * EnvTime::kMicrosToNanos);
        }

        void SetAllStart(NodeExecStatsInterface* stats) {
            if (!stats) return;
            // flush commonad queue first
            OPENGL_CALL(glFinish());
            stats->RecordExecutorStarted();
        }

        void SetOpStart(NodeExecStatsInterface* stats) {
            if (!stats) return;
            stats->RecordComputeStarted();
        }

        void SetOpEnd(NodeExecStatsInterface* stats) {
            if (!stats) return;
            stats->RecordComputeEnded();
        }

        void SetAllEnd(NodeExecStatsInterface* stats) {
            if (!stats) return;
            // make sure current node finished
            OPENGL_CALL(glFinish());
            stats->RecordExecutorEnded();
        }

        void SetOutput(NodeExecStatsInterface* stats, int slot, const Tensor* v) {
            if (!stats) return;
            stats->SetOutput(slot, v);
        }
    } // namespace nodestats

    FBOSession::~FBOSession(){
        // delete all tensors
    }

    void FBOSession::LoadGraph(const std::string file_path){
        auto model_proto = std::unique_ptr<::dlxnet::ModelProto>(
                new ::dlxnet::ModelProto);
        // load graph from disk
        CHECK(ReadProtoFromBinary(file_path.c_str(), model_proto.get()))
            <<"Load Graph "<<file_path <<"Failed";
        LoadGraph(*model_proto);
    }

    void FBOSession::LoadGraph(const ::dlxnet::ModelProto&& model_proto){
        LoadGraph(model_proto);
    }


    void FBOSession::LoadGraph(const ::dlxnet::ModelProto& model_proto){
        const uint64 start_time_usecs = env_->NowMicros();
        *model_=model_proto;
        // clear kernels first
        kernels_.clear();

        // LOG(INFO)<<"Write proto to Text";
        // WriteProtoToText("./demo.pbtxt", *model_);

        dlxnet::GraphProto graph = model_->graph();
        // create kernel and setup input and output for each node
        // Note that dont need to allocate memory due to lack of shape information
        total_tensors_.resize(graph.tensor_names_size());


        // build tensor_name -> tensor_index map
        for(int i=0;i<graph.tensor_names_size();++i){
            tensor_name_index_[graph.tensor_names(i)] = i;
        }


        Kernel* kernel;
        std::unique_ptr<Kernel> kernel_ptr;
        for(auto& node: graph.node()){
            kernel=nullptr;

            if(node.type()=="Input"){
                auto& input_attr = node.attr().input_attr();
                const int dims_size = input_attr.dims_size();
                CHECK_GE(dims_size, 2);
                for(int i=dims_size-2;i<dims_size;++i){
                    default_input_sizes_.emplace_back(input_attr.dims(i));
                }
                // LOG(WARNING)<<"Ignore Node Type: "<<node.type();
                continue;
            }
            KernelRegistry::Global()->CreateKernel(node.type(), &kernel, context_);
            if(kernel==nullptr){
                LOG(FATAL)<<"unsupported kernel name "<<node.type();
            }
            kernel_ptr.reset(kernel);
            kernel->Setup(node, this);

            kernels_.emplace_back(std::move(kernel_ptr));
        }
        finalized_ = false;
        graph_created_ = true;
        metrics::UpdateGraphBuildTime(env_->NowMicros() - start_time_usecs);
    }


    void FBOSession::Run(const NamedTensorList& inputs_cpu){
        profiler::TraceMe traceme("Run Time", 1, true);

        // session set up
        {
            profiler::TraceMe traceme("Setup Time", 1, true);
            Setup(inputs_cpu);
        }

        CHECK(finalized_)<<"Please Setup Session First";

        {
            profiler::TraceMe traceme("Computation Time", 1, true);
            for(int i=0;i<kernels_.size();++i){
                auto kernel = kernels_[i].get();
                if(CheckKernelReady(kernel)){
                    continue;
                }
                {
                    auto kernel_name= kernel->kernel_name();
                    auto kernel_type = kernel->kernel_type();
                    profiler::TraceMe traceme(kernel_name+"#"+kernel_type+"#",
                            1, true);
                    kernel->Compute();
                }

            }
        }

    }


    FBOSession::FBOSession(Context* context)
        :context_(context){
            // create vertex shader first
            model_ = new dlxnet::ModelProto;
            env_ = Env::Default();
            // reset context for current session
            context_->Reset();
        }

    void FBOSession::Setup(const NamedTensorList& inputs_cpu){
        CHECK(graph_created_)<<"No Graph Loaded!";
        // allocate memory for each tensor
        // so that dont need to allocate input and output tensors
        // for each kernel during computation

        // allocate memory for input tensor(device_tensor) first
        // TODO(breakpoint) add input-typed kernel
        for(auto input_iter=inputs_cpu.begin(); input_iter!=inputs_cpu.end(); ++input_iter){
            const Tensor* input_cpu = input_iter->second;
            const auto& tensor_name = input_iter->first;

            auto iter = tensor_name_index_.find(tensor_name);
            if(iter==tensor_name_index_.end()){
                LOG(FATAL)<<"tensor_name: "<<tensor_name<<" Cannot Find";
            }
            const int input_index = iter->second;

            if(!finalized_){
                // allocate memory in the first time
                total_tensors_[input_index].reset(
                        new Tensor(Tensor::DT_FLOAT, input_cpu->shape(),
                            Tensor::DEVICE_TEXTURE, FormatToStride4(input_cpu->dformat())));
            }
            // upload data, initialize input tensor
            context_->CopyCPUTensorToDevice(input_cpu, total_tensors_[input_index].get());
        }
        if(finalized_){return;}

        ready_.clear();
        ready_.resize(total_tensors_.size());
        std::fill(ready_.begin(), ready_.end(), false);

        for(int i=0;i<kernels_.size();++i){
            auto& kernel = kernels_[i];
            // clear input and output tensors
            kernel->input_tensors_.clear();
            kernel->output_tensors_.clear();
            LOG(INFO)<<"name: " << kernel->kernel_name()
                <<" type: "<<kernel->kernel_type();
            TensorShapeList output_shapes;
            for(int j=0; j<kernel->input_tensor_indexes_.size(); ++j){
                Tensor* input_tensor = total_tensors_[kernel->input_tensor_indexes_[j]].get();
                CHECK(input_tensor)<<"input tensor is uninitialized of kernel index: "<<i;
                kernel->input_tensors_.emplace_back(input_tensor);
            }
            // build kernel here before all tensors allocated
            kernel->SelectKernel(kernel->input_tensors_);
            // setup program for each kernel here
            kernel->SetupProgram(context_->CreateProgram(kernel->kernel_fname(),
                        kernel->build_options()));

            // infer output shapes from input shapes
            // Note that use input tensor as arg instead of input shapes
            // we need dformat(like nhwc) info to derminate the output shape no only the input shape.
            // kernel->InferOutputShape(input_tensors, output_shapes);
            kernel->InferOutputShape(kernel->input_tensors_, output_shapes);

            CHECK_GT(output_shapes.size(), 0);

            // allocate memory for each output tensors according to their shapes
            for(int j=0;j<output_shapes.size();++j){
                auto dformat = kernel->GetOutputDFormat(j);
                auto output_tensor = new Tensor(Tensor::DT_FLOAT, output_shapes[j],
                        Tensor::DEVICE_TEXTURE, dformat);
                total_tensors_[kernel->output_tensor_indexes_[j]].reset(output_tensor);

                kernel->output_tensors_.emplace_back(output_tensor);
            }



            if(CheckKernelReady(kernel.get())){
                // precompute kernel, not only used for constant kernel
                kernel->Compute();
                OPENGL_CHECK_ERROR;
                UpdateKernelReady(kernel.get());
            }

            // log kernel info after kernel finalized
            DLOG(INFO)<<kernel->DebugString();
        }
        finalized_ = true;
        OPENGL_CHECK_ERROR;
    }

    bool FBOSession::CheckKernelReady(const Kernel* kernel){
        // if some kernel is so special, e.g, shape kernel,
        // force it to execute
        if(kernel->ForceReady()){
            return true;
        }
        // find all precondition
        for(auto tensor_id: kernel->input_tensor_indexes_){
            if(not ready_[tensor_id]){
                return false;
            }
        }
        return true;
    }


    void FBOSession::UpdateKernelReady(const Kernel* kernel){
        for(auto tensor_id: kernel->output_tensor_indexes_){
            ready_[tensor_id]=true;
        }
    }

    void FBOSession::GetOutputs(const TensorNameList& output_names,
            const StringList& output_dformats, TensorList* outputs){
        profiler::TraceMe traceme("Output Time", 1, true);
        CHECK_EQ(output_names.size(), output_dformats.size());
        const int num_outputs = output_names.size();
        // outputs->clear();
        // outputs->reserve(output_names.size());
        if(outputs->size()<num_outputs){
            outputs->resize(num_outputs);
        }

        int index = 0;
        for(int i=0;i<output_names.size();++i){
            auto tensor_name = output_names[i];
            auto gpu_tensor = FindTensorByName(tensor_name);
            auto dformat_str = output_dformats[index++];
            DataFormat dformat = StrToFormat(dformat_str);
            if(outputs->at(i)==nullptr){
                outputs->at(i) = new Tensor(Tensor::DT_FLOAT, gpu_tensor->shape(),
                        Tensor::HOST_MEMORY, dformat);
            }else{
                // check its shape, it should be same with gpu tensor
            }
            context_->CopyDeviceTensorToCPU(gpu_tensor, outputs->at(i));
        }
    }

    std::string FBOSession::DebugString()const{
        std::string ret_str;
        ret_str+="ModelProto: ";
        ret_str+=model_->DebugString();
        ret_str+="\n";
        return ret_str;
    }

    Tensor* FBOSession::FindTensorByName(const std::string& tensor_name){
        auto iter = tensor_name_index_.find(tensor_name);
        if(iter==tensor_name_index_.end()){
            LOG(FATAL)<<"tensor_name: "<<tensor_name<<" Cannot Find";
        }

        const int tensor_index = tensor_name_index_[tensor_name];
        return FindTensorById(tensor_index);
    }

    Tensor* FBOSession::FindTensorById(const int id){
        CHECK_LT(id, total_tensors_.size());
        return total_tensors_[id].get();
    }
}//namespace opengl
