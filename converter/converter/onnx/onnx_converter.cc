#include <string>
#include <unordered_map>

#include "onnx/onnx_converter.h"
#include "core/utils.h"
#include "onnx/onnx_utils.h"
#include "core/op_converter.h"



void ONNXConverter::Reset(const ConverterConfig config){
    converter_config_ = config;
    model_ = new dlxnet::ModelProto;
    model_->set_producer_name("ONNX");
    model_->set_version("0.1");
    model_->set_doc_string("ignored");
}

void ONNXConverter::Run(){
    // load onnx proto
    onnx::ModelProto model_proto;
    bool success = utils::ONNXReadProtoFromBinary(converter_config_.src_model_path.c_str(), &model_proto);
    CHECK(success)<<"load onnx model failed!";
    const auto& graph_proto = model_proto.graph();
    const int node_counts = graph_proto.node_size();

    LOG(INFO)<<"node_counts: "<<node_counts;

    // The goal is to populate model proto

    // populate graph
    dlxnet::GraphProto* graph = model_->mutable_graph();
    // map from tensor name to tensor index
    std::unordered_map<std::string, int> total_tensor_names;
    // map from onnx nodes to dlcl nodes,
    // note that dlcl nodes also include constant node
    std::unordered_map<const onnx::NodeProto*, dlxnet::NodeProto*> onnx_dlcl_map;

    // insert input tensor to total_tensor_names first
    // map input_name to input tensor info
    std::unordered_map<std::string, const onnx::ValueInfoProto*> input_tensor_map;
    const int input_tensor_count = graph_proto.input_size();
    for(int i=0;i<input_tensor_count;++i){
        // add tensor name to map and graph proto at the sametime
        auto& tensor_name = graph_proto.input(i).name();
        total_tensor_names.insert({tensor_name, total_tensor_names.size()});
        input_tensor_map[tensor_name] = &graph_proto.input(i);
        graph->add_tensor_names(tensor_name);
    }

    // set input node first
    for(const auto& iter: total_tensor_names){
        auto& input_name = iter.first;
        auto& input_index = iter.second;
        dlxnet::NodeProto* node_ptr = graph->add_node();
        node_ptr->set_name(iter.first);
        node_ptr->set_type("Input");
        node_ptr->add_output_index(input_index);
        ::dlxnet::InputAttribute* input_attr = node_ptr->mutable_attr()
                    ->mutable_input_attr();

        // find input info from input_tensor_map
        auto it = input_tensor_map.find(input_name);
        const auto& tensorInfo = it->second->type().tensor_type();
        const int input_dims_size = tensorInfo.shape().dim_size();
        for(int i=0;i<input_dims_size;++i){
            input_attr->add_dims(tensorInfo.shape().dim(i).dim_value());
        }
    }

    // constant tensor map
    // help to find const tensor more quickly
    std::unordered_map<std::string, const onnx::TensorProto*>
        constant_tensor_map;
    const int constant_tensor_count = graph_proto.initializer_size();
    for(int i=0;i<constant_tensor_count;++i){
        auto& initializer = graph_proto.initializer(i);
        constant_tensor_map.insert({initializer.name(), &initializer});
    }


    for(int i=0;i<node_counts;i++){
        // dispatch each node handlers
        const auto& node_proto = graph_proto.node(i);
        const auto& op_type = node_proto.op_type();
        VLOG(1)<<"op_type: "<<op_type;

        // find op according to its name
        auto op_conveter_registry = Registry<OpConverter>::Global();
        OpConverter* op_converter=nullptr;
        op_conveter_registry->LookUp(op_type, &op_converter);
        if(op_type=="Concat"){
            int a = 10;
        }
        if(op_converter==nullptr){
            LOG(FATAL)<<"Cannot find Type: "<<op_type;
        }

        // handle with its input first
        // check if they contain constant tensor or not
        for(int j=0;j<node_proto.input_size();++j){
            auto& input_name = node_proto.input(j);
            // find input in constant map and total_tensor_names first
            if(total_tensor_names.find(input_name)!=total_tensor_names.end()){
                // already inserted, skip it
                continue;
            }
            auto iter = constant_tensor_map.find(input_name);
            if(iter!=constant_tensor_map.end()){
                // constant tensor
                dlxnet::NodeProto* node_ptr = graph->add_node();
                node_ptr->set_name(iter->first);
                node_ptr->set_type("Const");
                int output_tensor_index = total_tensor_names.size();
                node_ptr->add_output_index(output_tensor_index);

                // convert data

                dlxnet::TensorProto* tensor = node_ptr->mutable_attr()
                    ->mutable_const_attr()->mutable_value();
                MakeTensorFromProto(*iter->second, tensor);

                op_converter->SetTensorInfo(tensor, j);

                total_tensor_names.insert({input_name, output_tensor_index});
                graph->add_tensor_names(input_name);
            }
        }

        // then handle node according to its type
        dlxnet::NodeProto* node_ptr = graph->add_node();
        // set node name with the name of the first output
        node_ptr->set_name(node_proto.output(0));

        node_ptr->set_type(node_proto.op_type());
        // populate node
        op_converter->Run(node_ptr, &node_proto);

        // add map between onnx and dlcl
        onnx_dlcl_map.insert({&node_proto, node_ptr});
        // }

        // insert all output tensors to total_tensor_names
        for(int j=0;j<node_proto.output_size();++j){
            auto& output_name = node_proto.output(j);
            total_tensor_names.insert({output_name, total_tensor_names.size()});
            graph->add_tensor_names(output_name);
        }
}

// set input index and output index for each node op
// due to all tensor has already been inserted to total_tensor_names
// so the index is immutable.
// Note that no need to handle constant node again here
// because they are already done
for(int i=0;i<node_counts;++i){
    const auto& node_proto = graph_proto.node(i);
    // find its accord node in dlcl
    dlxnet::NodeProto* dlcl_node = onnx_dlcl_map[&node_proto];
    CHECK_NOTNULL(dlcl_node);
    // set input index
    for(int j=0;j<node_proto.input_size();++j){
        auto& input_name = node_proto.input(j);
        if(input_name==""){
            LOG(WARNING)<<"input name is empty, maybe need to check it";
        }else{
            auto iter = total_tensor_names.find(input_name);
            CHECK(iter!=total_tensor_names.end())<<
                "Cannot find the input tensor in total_tensor_names";
            dlcl_node->add_input_index(iter->second);
        }
    }

    // set output index
    for(int j=0;j<node_proto.output_size();++j){
        auto& output_name = node_proto.output(j);
        auto iter = total_tensor_names.find(output_name);
        CHECK(iter!=total_tensor_names.end())<<
            "Cannot find the output tensor in total_tensor_names";
        dlcl_node->add_output_index(iter->second);
    }
}

// set total tensor name and output names in graph
// output names
for(int i=0;i<graph_proto.output_size();++i){
    graph->add_output_names(graph_proto.output(i).name());
}

// input names
for(int i=0;i<graph_proto.input_size();++i){
    graph->add_input_names(graph_proto.input(i).name());
}

LOG(INFO)<<"ONNXConverter Done!";
}


