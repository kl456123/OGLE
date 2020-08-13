#include <memory>
#include <string>
#include <algorithm>
#include <glog/logging.h>

#include "graph/graph.h"


namespace graph{
    namespace{

        void TopologicalSort(const Graph* graph, std::vector<int>* sorted_node_indexes,
                const std::set<std::string>& target_node_names){
            std::set<int> ready;
            std::vector<int> pending;

            for(int i=0;i<graph->num_node_ids();++i){
                const Node* node = graph->FindNodeId(i);
                // The current Node in Index is already freed
                if(node==nullptr){
                    pending.emplace_back(0);
                    continue;
                }

                pending.emplace_back(node->num_inputs());
                if(node->num_inputs()==0){
                    ready.insert(i);
                }
            }
            while(ready.size()){
                int o = *ready.begin();
                ready.erase(ready.begin());
                const Node* node = graph->FindNodeId(o);

                // skip unused op when satisfing both conditions
                // 1. dead
                // 2. not be used by output
                if(node->out_edges().size()==0&&node->num_inputs()==0
                        && (target_node_names.find(node->name())==target_node_names.end())){continue;}
                sorted_node_indexes->emplace_back(o);

                // update succeed nodes
                for(const Edge* edge: node->out_edges()){
                    auto node_id = edge->dst()->id();
                    if(node_id<0){continue;}
                    CHECK_GT(pending[node_id], 0);
                    pending[node_id]--;
                    if(pending[node_id]==0){
                        ready.insert(node_id);
                    }
                }
            }
        }
    }
    struct NodeProperties{
        NodeProperties(::dlxnet::NodeProto node_def, const std::string op_type)
            :node_def(std::move(node_def)), op_type(op_type){}

        ::dlxnet::NodeProto node_def;
        const std::string op_type;
    };

    Node::Node()
        : id_(-1),
        class_(NC_UNINITIALIZED),
        props_(nullptr){}
    // note here we use num of input edge to get the num of input tensor slot
    // due to it is one to one mapping for input for now
    // but it is wrong for output due to prev node can feed tensor to multiple succeed nodes
    int32_t Node::num_outputs() const { return props_->node_def.output_index().size(); }
    int32_t Node::num_inputs() const { return in_edges_.size(); }
    const std::string& Node::name() const { return props_->node_def.name(); }
    void Node::set_name(std::string name) { return props_->node_def.set_name(name); }
    const std::string& Node::type_string() const { return props_->node_def.type(); }
    const ::dlxnet::NodeProto& Node::def() const { return props_->node_def; }
    ::dlxnet::NodeProto& Node::def() { return props_->node_def; }

    std::string Node::DebugString() const {
        std::stringstream ret;
        ret<<"{name:'"<<name()<<"' id:"<<id_
            <<" type: "<<type_string()<< "}";
        return ret.str();
    }

    void Node::Initialize(int id,  std::shared_ptr<NodeProperties> props){
        DCHECK_EQ(id_, -1);
        DCHECK(in_edges_.empty());
        DCHECK(out_edges_.empty());
        id_ = id;

        props_ = std::move(props);
        // Initialize the class_ based on the type string
        // TODO(breakpoint) specify which node class to be used for op
        // class_ = GetNodeClassForOp(props_->node_def.op());
    }

    void Node::output_edge(int idx, const Edge** e)const{
        if (idx < 0 || idx >= num_outputs()) {
            LOG(FATAL)<<"Invalid output_edge index: "<< idx<< ", Node "<<
                name()<< " only has "<< num_outputs()<<
                " outputs.";
        }

        for (const Edge* edge : out_edges()) {
            if (edge->src_output() == idx) {
                *e = edge;
                return ;
            }
        }
    }

    void Node::Clear() {
        in_edges_.clear();
        out_edges_.clear();
        id_ = -1;
        class_ = NC_UNINITIALIZED;
        props_.reset();
    }

    void Node::input_edge(int idx, const Edge** e) const {
        if (idx < 0 || idx >= num_inputs()) {
            LOG(FATAL)<<"Invalid input_edge index: "<< idx<< ", Node "<<
                name()<< " only has "<< num_inputs()<<
                " inputs.";
        }

        // This does a linear search over the edges.  In the common case,
        // the number of elements is small enough that this search isn't
        // expensive.  Should it become a bottleneck, one can make an
        // optimization where, if the number of edges is small, we use
        // linear iteration, and if the number of edges is large, we perform
        // an indexing step during construction that keeps an array of Edges
        // indexed by pointer.  This would keep the size of each Node small
        // in the common case but make this function faster when the number
        // of edges is large.
        for (const Edge* edge : in_edges()) {
            if (edge->dst_input() == idx) {
                *e = edge;
                return ;
            }
        }

        LOG(FATAL)<<"Could not find input edge "<< idx<< " for "<< name();
    }

    Graph::Graph(){
        // specify opset for the graph
        // where "opset" means system defined operators
    }

    Node* Graph::AddNode(::dlxnet::NodeProto node_def){
        // TODO(breakpoint) use kernel or kernel_type?
        const std::string op_type = node_def.type();
        Node* node = AllocateNode(std::make_shared<NodeProperties>(
                    std::move(node_def), op_type));
        return node;
    }

    const Edge* Node::input_edge(int idx)const{
        const Edge* e;
        input_edge(idx, &e);
        return e;
    }

    const Edge* Node::output_edge(int idx)const{
        const Edge* e;
        output_edge(idx, &e);
        return e;
    }

    void Graph::ReleaseNode(Node* node) {
        nodes_[node->id()] = nullptr;
        --num_nodes_;
        node->Clear();
    }

    void Graph::RemoveNode(Node* node){
        // Remove any edges involving this node.
        for (const Edge* e : node->in_edges_) {
            CHECK_EQ(e->src_->out_edges_.erase(e), size_t{1});
            edges_[e->id_] = nullptr;
            --num_edges_;
        }
        node->in_edges_.clear();
        for (const Edge* e : node->out_edges_) {
            CHECK_EQ(e->dst_->in_edges_.erase(e), size_t{1});
            edges_[e->id_] = nullptr;
            --num_edges_;
        }
        node->out_edges_.clear();
        ReleaseNode(node);
    }

    const Edge* Graph::AddEdge(Node* source, int x, Node* dest, int y){
        // check any edge free exist
        Edge* e = nullptr;
        e = new Edge;

        // populate edge, add it to src node and dst node,
        // then add it to graph
        e->id_ = edges_.size();
        e->src_ = source;
        e->dst_ = dest;
        e->src_output_ = x;
        e->dst_input_ = y;
        CHECK(source->out_edges_.insert(e).second);
        CHECK(dest->in_edges_.insert(e).second);
        edges_.push_back(e);
        ++num_edges_;
        return e;
    }

    std::string Graph::NewName(std::string prefix){
        return "";
    }

    void Graph::RemoveEdge(const Edge* edge){
    }

    Node* Graph::AllocateNode(std::shared_ptr<NodeProperties> props){
        Node* node;
        node = new Node;

        // initialize node here
        const int id = nodes_.size();
        node->graph_ = this;
        node->Initialize(id, std::move(props));
        nodes_.push_back(node);
        ++num_nodes_;
        return node;
    }

    void Graph::ToGraphDef(::dlxnet::GraphProto* graph_def) const{
        graph_def->Clear();
        graph_def->mutable_node()->Reserve(std::max(1, num_node_ids()));
        std::vector<const Edge*> inputs;  // Construct this outside the loop for speed.

        // map from tensor name to tensor index
        std::unordered_map<std::string, int> total_tensor_names;
        // topological sort first and remove unused node at the same time
        std::vector<int>  sorted_node_indexes;

        // TODO(breakpoint) should also use input_names here ?
        std::set<std::string> target_node_names;
        for(auto& name: output_names_){
            target_node_names.insert(name);
        }
        TopologicalSort(this, &sorted_node_indexes, target_node_names);


        for(auto i: sorted_node_indexes){
            const Node* node = FindNodeId(i);
            // The current Node in Index is already freed
            if(node==nullptr){
                continue;
            }

            auto node_def = graph_def->add_node();
            // set node_def from scratch
            node_def->mutable_attr()->CopyFrom(node->def().attr());
            node_def->set_name(node->name());
            node_def->set_type(node->type_string());
            node_def->set_doc_string(node->def().doc_string());

            auto input_index = node_def->mutable_input_index();
            input_index->Resize(node->num_inputs(), 0);
            // set input index
            for(auto edge: node->in_edges()){
                node->input_edge(0);
                // const Edge* edge = node->input_edge(i);
                // make sure all input prepare
                const Node* src = edge->src();
                // consider node name as output tensor name
                // so only one output tensor is supported now.
                // TODO(breakpoint) use node_name:out_index as tensor_name
                auto iter = total_tensor_names.find(src->name());
                CHECK(iter!=total_tensor_names.end())<<"Input Tensor: "
                    <<src->name()<<" Cannot be Prepared";
                input_index->Set(edge->dst_input(), iter->second);
                // node_def->mutable_input_index()();
            }
            // Input Node Type
            if(node->type_string()=="Input"){
                graph_def->add_input_names(node_def->name());
            }

            // all endpoints in graph considered as output node
            if(node->num_outputs()==0){
                graph_def->add_output_names(node_def->name());
            }else{
                // only single output slot supported now
                CHECK_EQ(node->num_outputs(), 1)<<node->DebugString();
            }

            const int tensor_index = total_tensor_names.size();
            total_tensor_names.insert({node_def->name(), tensor_index});
            graph_def->add_tensor_names(node_def->name());
            node_def->add_output_index(tensor_index);

            // check for some type
            if(node->type_string()=="Const"|| node->type_string()=="Input"){
                // only one output port for constant node
                CHECK_EQ(node_def->output_index_size(), 1)<<" In Index "<<i
                    <<" Name: "<<node_def->name();
                CHECK_EQ(node_def->input_index_size(), 0)<<" In Index "<<i
                    <<" Name: "<<node_def->name();
            }

            if(node->type_string()=="Conv"){
                CHECK_EQ(node_def->output_index_size(), 1);
                CHECK(node_def->input_index_size()== 2
                        ||node_def->input_index_size()== 3);
            }

            if(node->type_string()=="BatchNormalization"){
                CHECK_EQ(node_def->input_index_size(), 5);
            }

            if(node->type_string()=="MaxPool"){
                CHECK_EQ(node_def->output_index_size(), 1)<<" In Index "<<i
                    <<" Name: "<<node_def->name();
                CHECK_EQ(node_def->input_index_size(), 1)<<" In Index "<<i
                    <<" Name: "<<node_def->name();
            }
        }
    }


    void Graph::SetupInputAndOutputNames(const ::dlxnet::GraphProto& graph_def){
        for(const auto& name: graph_def.input_names()){
            input_names_.emplace_back(std::move(name));
        }

        for(const auto& name: graph_def.output_names()){
            output_names_.emplace_back(std::move(name));
        }
    }


    Graph::~Graph(){}
}
