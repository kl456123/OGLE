#include<fstream>
#include <memory>
#include <glog/logging.h>

#include "core/converter.h"
#include "graph/graph_constructor.h"



void Converter::Save(std::string checkpoint_path){
    CHECK_NOTNULL(model_);
    // save to ckpt path
    std::fstream output(checkpoint_path, std::ios::out
            | std::ios::trunc | std::ios::binary);
    model_->SerializeToOstream(&output);
    LOG(INFO)<<"Save to "<<checkpoint_path<<" Done!";
}


std::string Converter::DebugString()const{
    std::string ret_str;
    ret_str+="ConverterConfig: ";
    ret_str+=converter_config_.src_model_path;
    ret_str+="->";
    ret_str+=converter_config_.dst_model_path;
    ret_str+="\n";

    ret_str+="ModelInfo: ";
    ret_str+=model_->DebugString();
    return ret_str;
}

void Converter::Optimize(const Optimizer* optimizer){
    // convert ModelProto to DAG-based Graph
    auto base_graph = std::unique_ptr<graph::Graph>(new graph::Graph());
    graph::ConvertGraphDefToGraph(model_->graph(), base_graph.get());
    optimizer->Optimize(base_graph.get());
    base_graph->ToGraphDef(model_->mutable_graph());

    LOG(INFO)<<"The information of model proto after optimized:"
        << "Node Nums: "<<model_->graph().node_size();
}
