#include <glog/logging.h>
#include <vector>
#include <cmath>
#include <unordered_set>

#include "optimizers/remapper.h"
#include "optimizers/op_types.h"



namespace optimizer{
    namespace{
        typedef std::vector<graph::Node*> NodeList;
        constexpr int kMissingIndex = -1;
        // matched pattern
        struct ContractionWithBatchNormAndActivation{
            ContractionWithBatchNormAndActivation()=default;
            ContractionWithBatchNormAndActivation(int contraction,
                    int batchnorm, int activation)
                :batchnorm(batchnorm), contraction(contraction),
                activation(activation){}
            int batchnorm=kMissingIndex;
            int contraction=kMissingIndex;
            int activation=kMissingIndex;
        };

        struct ContractionWithBatchNorm{
            ContractionWithBatchNorm()=default;
            ContractionWithBatchNorm(int contraction, int batchnorm)
                :batchnorm(batchnorm), contraction(contraction){}
            int batchnorm=kMissingIndex;
            int contraction=kMissingIndex;
        };

        bool FindContractionWithBatchNorm(graph::Graph* graph,
                int node_index,  ContractionWithBatchNorm* matched){
            auto node = graph->FindNodeId(node_index);
            if(!IsBatchnorm(*node)){
                return false;
            }
            auto contraction_node = node->input_edge(0)->src();
            if(!IsConv2D(*contraction_node)){
                return false;
            }
            ContractionWithBatchNorm pattern{contraction_node->id(), node_index};
            *matched = pattern;
            return true;
        }

        bool FindContractionWithBatchNormAndActivation(graph::Graph* graph,
                int node_index,  ContractionWithBatchNormAndActivation* matched){
            // reverse traverse
            auto node = graph->FindNodeId(node_index);
            if(!IsActivation(*node))return false;

            auto batchnorm_node = node->input_edge(0)->src();
            ContractionWithBatchNorm base;
            if(!FindContractionWithBatchNorm(graph, batchnorm_node->id(), &base)){
                return false;
            }
            ContractionWithBatchNormAndActivation pattern{base.contraction,
                base.batchnorm, node->id()};
            *matched = pattern;
            return true;
        }

        graph::Node* MakeNode(graph::Graph* graph, const std::vector<int> dims, int node_index){
            // add const node to node_index
            auto dst_node = graph->FindNodeId(node_index);
            const int input_index = dst_node->num_inputs();
            std::string node_name = dst_node->name()+":"+std::to_string(input_index);
            ::dlxnet::NodeProto node_proto;
            node_proto.set_name(node_name);
            node_proto.set_type("Const");
            // it equals to the node id in the graph
            node_proto.add_output_index(graph->num_node_ids());
            // set value
            dlxnet::TensorProto* tensor = node_proto.mutable_attr()
                ->mutable_const_attr()->mutable_value();
            int num_elements=1;
            for(auto dim:dims){
                tensor->add_dims(dim);
                num_elements*=dim;
            }
            // allocate it here
            tensor->mutable_float_data()->Resize(num_elements, 0);
            // set tensor info
            tensor->set_data_type(dlxnet::TensorProto::FLOAT32);
            tensor->set_data_format(dlxnet::TensorProto::ANY);
            tensor->set_target_data_format(dlxnet::TensorProto::ANY4);

            auto src_node = graph->AddNode(node_proto);
            graph->AddEdge(src_node, 0, dst_node, input_index);
            return src_node;
        }

        void MergeBatchNormToConvolution(graph::Graph* graph,
                ContractionWithBatchNorm& matched, std::vector<bool>* nodes_to_delete){
            auto contraction_node  = graph->FindNodeId(matched.contraction);
            auto batchnorm_node = graph->FindNodeId(matched.batchnorm);
            // find batchnorm params
            // gamma, beta, mean, var
            auto gamma_node = batchnorm_node->input_edge(1)->src();
            auto beta_node = batchnorm_node->input_edge(2)->src();
            auto mean_node = batchnorm_node->input_edge(3)->src();
            auto var_node = batchnorm_node->input_edge(4)->src();
            auto epsilon = batchnorm_node->def().attr().batchnorm_attr().epsilon();

            auto gamma_data = gamma_node->def().attr().const_attr().value().float_data();
            auto beta_data = beta_node->def().attr().const_attr().value().float_data();
            auto mean_data = mean_node->def().attr().const_attr().value().float_data();
            auto var_data = var_node->def().attr().const_attr().value().float_data();

            // get size
            auto dims = gamma_node->def().attr().const_attr().value().dims();
            int num_elements = 1;
            for(auto dim: dims){
                num_elements *= dim;
            }

            /// get conv2d params
            contraction_node->set_name(batchnorm_node->name());
            auto weight_node = contraction_node->input_edge(1)->src();
            bool use_bias = contraction_node->num_inputs()>2;
            graph::Node* bias_node;
            if(use_bias){
                bias_node = contraction_node->input_edge(2)->src();
            }else{
                // add bias to graph
                bias_node = MakeNode(graph, {num_elements},
                        contraction_node->id());
            }
            auto bias_data = bias_node->def().mutable_attr()->mutable_const_attr()
                ->mutable_value()->mutable_float_data()->mutable_data();
            // n_out, n_in, h, w
            auto weight_data = weight_node->def().mutable_attr()->mutable_const_attr()
                ->mutable_value()->mutable_float_data()->mutable_data();
            auto weight_dims = weight_node->def().attr().const_attr().value().dims();
            const int chw = weight_dims[1]*weight_dims[2]*weight_dims[3];

            // TODO(breakpoint) use parallel style
            for(int i=0; i<num_elements; ++i){
                auto scale = gamma_data[i]/std::sqrt(var_data[i]+epsilon);
                auto bias = -scale * mean_data[i] + beta_data[i];
                for(int j=0;j<chw;++j){
                    weight_data[i*chw+j] *=scale;
                }
                bias_data[i] = bias_data[i]*scale+bias;
            }
            (*nodes_to_delete)[matched.batchnorm] = true;
        }

        void MergeBatchNormActivationToConvolution(graph::Graph* graph,
                ContractionWithBatchNormAndActivation& matched,
                std::vector<bool>* nodes_to_delete){
            ContractionWithBatchNorm base{matched.contraction, matched.batchnorm};
            (*nodes_to_delete)[matched.activation] = true;
            auto contraction_node  = graph->FindNodeId(matched.contraction);
            auto act_node  = graph->FindNodeId(matched.activation);
            auto conv2d_attr = contraction_node->def().mutable_attr()->mutable_conv2d_attr();
            conv2d_attr->set_activation_type(act_node->type_string());

            if(act_node->type_string()== "Clip"){
                // clip
                auto clip_attr = act_node->def().attr().clip_attr();
                conv2d_attr->set_max(clip_attr.max());
            }else{
                // relu
                conv2d_attr->set_min(0);
                conv2d_attr->set_max(0);
            }


            MergeBatchNormToConvolution(graph, base, nodes_to_delete);
            contraction_node->set_name(act_node->name());
        }
    }


    void Remapper::Run(graph::Graph* graph){
        std::vector<bool> nodes_to_delete(graph->num_node_ids(), false);

        // travel in reverse order
        for(int i=graph->num_node_ids()-1;i>=0;--i){
            // find specified pattern from each node
            auto node = graph->FindNodeId(i);
            if(nodes_to_delete[i])continue;

            ContractionWithBatchNormAndActivation contraction_with_batchnorm_activation;
            ContractionWithBatchNorm contraction_with_batchnorm;


            if(FindContractionWithBatchNorm(graph, i, &contraction_with_batchnorm)){
                MergeBatchNormToConvolution(graph, contraction_with_batchnorm, &nodes_to_delete);
                continue;
            }

            if(FindContractionWithBatchNormAndActivation(graph, i,
                        &contraction_with_batchnorm_activation)){
                MergeBatchNormActivationToConvolution(graph, contraction_with_batchnorm_activation,
                        &nodes_to_delete);
                continue;
            }
        }

        for(int i=0;i<nodes_to_delete.size();++i){
            if(!nodes_to_delete[i])continue;
            auto node = graph->FindNodeId(i);

            auto src = node->input_edge(0)->src();
            auto src_index = node->input_edge(0)->src_output();

            std::vector<int> dst_indexes;
            std::vector<graph::Node*> dst_nodes;
            for(auto e: node->out_edges()){
                dst_nodes.emplace_back(e->dst());
                dst_indexes.emplace_back(e->dst_input());
            }
            // then remove current node
            graph->RemoveNode(node);

            for(int i=0; i<dst_indexes.size(); ++i){
                // finally reconnect prev node and next node
                graph->AddEdge(src, src_index, dst_nodes[i], dst_indexes[i]);
            }
        }
    }
    REGISTER_PASS_WITH_NAME(Remapper, "Remapper");
}
