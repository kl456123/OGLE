#include "optimizers/op_types.h"


namespace optimizer{
    bool IsConv2D(const graph::Node& node){return node.type_string()=="Conv";}
    bool IsActivation(const graph::Node& node){
        return node.type_string()=="Relu"
            || node.type_string()=="PRelu"
            || node.type_string()=="Clip";}
    bool IsBatchnorm(const graph::Node& node){
        return node.type_string()=="BatchNormalization";
    }
    bool IsBinary(const graph::Node& node){
        auto& type = node.type_string();
        return type=="Add" || type=="Subtract"
            ||type=="Multiple" || type=="Division";
    }
}//namespace optimizer
