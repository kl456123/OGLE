#include <glog/logging.h>

#include "core/optimizer.h"

Optimizer::Optimizer(){
}

void Optimizer::RegisterPass(std::string pass_name,
        OptimizationPass* pass){
    passes_.insert({pass_name, pass});
}

void Optimizer::LookUpPass(std::string pass_name,
        OptimizationPass** out)const{
    auto iter = passes_.find(pass_name);
    if(iter==passes_.end()){
        LOG(FATAL)<<"Cannot find Pass: "<<pass_name;
    }
    *out = iter->second;
}

void Optimizer::Optimize(graph::Graph* graph)const{
    std::vector<std::string> pass_names = {"Remapper"};
    LOG(INFO)<<"The information of graph before optimized:"
        << "Node Nums: "<<graph->num_nodes()
        << "Edge Nums: "<<graph->num_edges();
    // run all passes in order by default
    for(auto pass_name: pass_names){
        OptimizationPass* pass=nullptr;
        LookUpPass(pass_name, &pass);
        pass->Run(graph);
    }

    LOG(INFO)<<"The information of graph after optimized:"
        << "Node Nums: "<<graph->num_nodes()
        << "Edge Nums: "<<graph->num_edges();
}

/*static*/ Optimizer* Optimizer::Global(){
    static auto optimizer = new Optimizer();
    return optimizer;
}



