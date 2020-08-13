#ifndef GRAPH_GRAPH_CONSTRUCTOR_H_
#define GRAPH_GRAPH_CONSTRUCTOR_H_
#include "graph.h"

namespace graph{
    // fix some name
    typedef ::dlxnet::NodeProto NodeDef;
    typedef ::dlxnet::GraphProto GraphDef;

    // construct from empty graph(sink and source node)
    bool ConvertGraphDefToGraph(const GraphDef& gdef, Graph* g);
    bool ConvertGraphDefToGraph(GraphDef&& gdef, Graph* g);
}// namespace graph


#endif
