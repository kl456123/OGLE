#ifndef CONVERTER_CORE_OP_CONVERTER_H_
#define CONVERTER_CORE_OP_CONVERTER_H_
#include <glog/logging.h>

#include "core/registry.h"
#include "dlcl.pb.h"



class OpConverter{
    public:
        OpConverter();
        virtual ~OpConverter();

        // derived class implement this func
        virtual void Run(dlxnet::NodeProto* dst_node, const void* src_node)=0;

        // set tensor shape ,dformat and target dformat used in target device
        virtual void SetTensorInfo(dlxnet::TensorProto* dlcl_tensor, int tensor_index);
};


#define REGISTER_CLASS_OP(CLASS)   \
    REGISTER_CLASS(OpConverter, CLASS)

#define REGISTER_OP_WITH_NAME(CLASS, name)  \
    REGISTER_CLASS_BY_NAME(OpConverter, name, CLASS)


#define DECLARE_OP_CONVERTER(name)                                                      \
    class name##OpConverter: public OpConverter{                                        \
    public:                                                                             \
        name##OpConverter(){}                                                           \
        virtual ~name##OpConverter(){}                                                  \
        virtual void Run(dlxnet::NodeProto* dst_node, const void* src_node)override;    \
        virtual void SetTensorInfo(dlxnet::TensorProto* dlcl_tensor, int tensor_index); \
}





#endif
