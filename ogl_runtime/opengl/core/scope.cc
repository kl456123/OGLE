#include "opengl/core/scope.h"


namespace opengl{
    Scope::Scope():tensor_index_(0){
        // init dlxnet model
        model_ = new dlxnet::ModelProto;
        model_->set_producer_name("ONNX");
        model_->set_version("0.1");
        model_->set_doc_string("ignored");
    }
    Scope::~Scope(){
        delete model_;
    }

    ::dlxnet::NodeProto* Scope::AddNode(){
        return model_->mutable_graph()->add_node();
    }

    void Scope::AddInputName(const std::string& name){
        model_->mutable_graph()->add_input_names(name);
    }

    void Scope::AddOutputName(const std::string& name){
        model_->mutable_graph()->add_output_names(name);
    }

    int Scope::AddTensor(const std::string& tensor_name){
        // add tensor to graph
        // for now just increase the index of tensor
        model_->mutable_graph()->add_tensor_names(tensor_name);
        return tensor_index_++;
    }
}//namespace opengl
