#include "core/op_converter.h"
#include "onnx.pb.h"


DECLARE_OP_CONVERTER(BatchNorm);




void BatchNormOpConverter::SetTensorInfo(dlxnet::TensorProto* dlcl_tensor,
        int tensor_index){
    // CHECK_EQ(tensor_index, 0);
    CHECK_EQ(dlcl_tensor->dims_size(), 1);
    dlcl_tensor->set_target_data_format(dlxnet::TensorProto::ANY4);
    // const int n_out = dlcl_tensor->dims(0);
    // dlcl_tensor->clear_dims();
    // dlcl_tensor->add_dims(1);
    // dlcl_tensor->add_dims(n_out);
    // dlcl_tensor->add_dims(1);
    // dlcl_tensor->add_dims(1);
    dlcl_tensor->set_data_format(dlxnet::TensorProto::ANY);
}

void BatchNormOpConverter::Run(dlxnet::NodeProto* dst_node, const void* src_node){
    dlxnet::BatchNormAttribute* dst_attr = dst_node->mutable_attr()->mutable_batchnorm_attr();
    const auto src_node_onnx = reinterpret_cast<const onnx::NodeProto*>(src_node);
    for(int i=0;i<src_node_onnx->attribute_size();i++){
        const onnx::AttributeProto& attr = src_node_onnx->attribute(i);
        // 1. epsilon
        // 2. momentum
        if(attr.name()=="epsilon"){
            dst_attr->set_epsilon(attr.f());
        }else if(attr.name()=="momentum"){
            dst_attr->set_momentum(attr.f());
        }
    }
}

REGISTER_OP_WITH_NAME(BatchNormOpConverter, "BatchNormalization");
