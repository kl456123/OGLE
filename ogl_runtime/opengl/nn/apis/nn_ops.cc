#include "opengl/nn/apis/nn_ops.h"
#include "opengl/core/tensor.h"


namespace opengl{
    int AddConstNode(Scope* scope, const std::string&  name, const Tensor* cpu_tensor){
        auto node_ptr = scope->AddNode();
        node_ptr->set_name(name);
        node_ptr->set_type("Const");
        int tensor_id = scope->AddTensor(name);
        node_ptr->add_output_index(tensor_id);
        dlxnet::TensorProto* tensor_proto = node_ptr->mutable_attr()
            ->mutable_const_attr()->mutable_value();

        cpu_tensor->AsProto(tensor_proto);

        return tensor_id;
    }
    int AddConstNode(Scope* scope, const std::string&  name,
            const std::vector<int>& shape, DataFormat dformat, DataFormat src_dformat){
        auto node_ptr = scope->AddNode();
        node_ptr->set_name(name);
        node_ptr->set_type("Const");
        int tensor_id = scope->AddTensor(name);
        node_ptr->add_output_index(tensor_id);
        dlxnet::TensorProto* dlcl_tensor = node_ptr->mutable_attr()
            ->mutable_const_attr()->mutable_value();

        int num_elements = 1;
        for(auto dim: shape){
            dlcl_tensor->add_dims(dim);
            num_elements  *= dim;
        }
        for(int j=0;j<num_elements;++j){
            dlcl_tensor->add_float_data(1.0*random()/RAND_MAX);
        }
        // set tensor
        dlcl_tensor->set_data_type(dlxnet::TensorProto::FLOAT32);
        dlcl_tensor->set_target_data_format(dformat);
        dlcl_tensor->set_data_format(src_dformat);
        return tensor_id;

    }

    // TODO(breakpoint) change id to NodeOut class to store shape info
    // can use shape info do some things to validate
    int AddConvNode(Scope* scope, const std::string&  name, std::vector<int> input_ids,
            const Conv2dParams& conv2d_params){
        auto dlcl_node = scope->AddNode();
        // set node name with the name of the first output
        dlcl_node->set_name(name);

        dlcl_node->set_type("Conv");
        dlxnet::Conv2dAttribute* dst_attr = dlcl_node->mutable_attr()->mutable_conv2d_attr();
        for(int i=0;i<2;++i){
            dst_attr->add_kernel_shape(conv2d_params.kernel_size);
        }
        for(int i=0;i<2;++i){
            dst_attr->add_strides(conv2d_params.stride);
        }
        for(int i=0;i<4;++i){
            dst_attr->add_pads(conv2d_params.padding);
        }
        for(auto tensor_id:input_ids){
            dlcl_node->add_input_index(tensor_id);
        }

        // set groups and dilations
        dst_attr->set_group(conv2d_params.groups);
        CHECK_GT(conv2d_params.dilation, 0);
        dst_attr->add_dilations(conv2d_params.dilation);

        // both node name and tensor name are the same
        int tensor_id = scope->AddTensor(name);
        dlcl_node->add_output_index(tensor_id);
        return tensor_id;
    }

    int AddInputNode(Scope* scope, std::string name){
        // add node
        scope->AddInputName(name);
        // add tensor
        return scope->AddTensor(name);
    }

    int AddShapeNode(Scope* scope, std::string name, std::vector<int> input_ids){
        auto dlcl_node = scope->AddNode();
        dlcl_node->set_name(name);
        dlcl_node->set_type("Shape");
        int tensor_id = scope->AddTensor(name);
        for(auto tensor_id:input_ids){
            dlcl_node->add_input_index(tensor_id);
        }
        dlcl_node->add_output_index(tensor_id);
        return tensor_id;
    }

    int AddConcatNode(Scope* scope, const std::string&  name, std::vector<int> input_ids,
            const ConcatParams& concat_params){
        auto dlcl_node = scope->AddNode();
        dlcl_node->set_name(name);
        dlcl_node->set_type("Concat");
        int tensor_id = scope->AddTensor(name);
        for(auto tensor_id:input_ids){
            dlcl_node->add_input_index(tensor_id);
        }
        dlcl_node->add_output_index(tensor_id);
        auto concat_attr = dlcl_node->mutable_attr()
            ->mutable_concat_attr();
        concat_attr->set_axis(concat_params.axis);
        return tensor_id;
    }

    int AddReshapeNode(Scope* scope, const std::string&  name, std::vector<int> input_ids){
        auto dlcl_node = scope->AddNode();
        dlcl_node->set_name(name);
        dlcl_node->set_type("Reshape");
        int tensor_id = scope->AddTensor(name);
        for(auto tensor_id:input_ids){
            dlcl_node->add_input_index(tensor_id);
        }
        dlcl_node->add_output_index(tensor_id);
        return tensor_id;
    }

    int AddTransposeNode(Scope* scope, const std::string& name, std::vector<int> input_ids,
            const TransposeParams& trans_params){
        auto dlcl_node = scope->AddNode();
        dlcl_node->set_name(name);
        dlcl_node->set_type("Transpose");
        int tensor_id = scope->AddTensor(name);
        for(auto tensor_id:input_ids){
            dlcl_node->add_input_index(tensor_id);
        }
        dlcl_node->add_output_index(tensor_id);

        auto trans_attr = dlcl_node->mutable_attr()
            ->mutable_transpose_attr();
        for(auto perm:trans_params.perm){
            trans_attr->add_perm(perm);
        }
        return tensor_id;
    }

    int AddClipNode(Scope* scope, const std::string& name, std::vector<int> input_ids,
            const ClipParams& clip_params){
        auto dlcl_node = scope->AddNode();
        dlcl_node->set_name(name);
        dlcl_node->set_type("Clip");
        int tensor_id = scope->AddTensor(name);
        for(auto tensor_id:input_ids){
            dlcl_node->add_input_index(tensor_id);
        }
        dlcl_node->add_output_index(tensor_id);

        auto clip_attr = dlcl_node->mutable_attr()
            ->mutable_clip_attr();
        clip_attr->set_min(clip_params.min);
        clip_attr->set_max(clip_params.max);
        return tensor_id;
    }

    int AddFlattenNode(Scope* scope, const std::string& name, std::vector<int> input_ids,
            const FlattenParams& flatten_params){
        auto dlcl_node = scope->AddNode();
        dlcl_node->set_name(name);
        dlcl_node->set_type("Flatten");
        int tensor_id = scope->AddTensor(name);
        for(auto tensor_id:input_ids){
            dlcl_node->add_input_index(tensor_id);
        }
        dlcl_node->add_output_index(tensor_id);

        auto flatten_attr = dlcl_node->mutable_attr()
            ->mutable_flatten_attr();
        flatten_attr->set_axis(flatten_params.axis);
        return tensor_id;
    }

    int AddGemmNode(Scope* scope, const std::string& name, std::vector<int> input_ids,
            const GemmParams& gemm_params){
        auto dlcl_node = scope->AddNode();
        dlcl_node->set_name(name);
        dlcl_node->set_type("Gemm");
        int tensor_id = scope->AddTensor(name);
        for(auto tensor_id:input_ids){
            dlcl_node->add_input_index(tensor_id);
        }
        dlcl_node->add_output_index(tensor_id);

        auto gemm_attr = dlcl_node->mutable_attr()
            ->mutable_gemm_attr();
        gemm_attr->set_alpha(gemm_params.alpha);
        gemm_attr->set_beta(gemm_params.beta);
        gemm_attr->set_transb(gemm_params.transb);
        return tensor_id;
    }
}//namespace opengl
