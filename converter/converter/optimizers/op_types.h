#ifndef OPTIMIZERS_OP_TYPES_H_
#define OPTIMIZERS_OP_TYPES_H_
#include "graph/graph.h"

namespace optimizer{
    bool IsConv2D(const graph::Node& node);
    bool IsActivation(const graph::Node& node);
    bool IsBatchnorm(const graph::Node& node);
    bool IsBinary(const graph::Node& node);
}//namespace optimizer

#endif
