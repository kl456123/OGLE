#include <unordered_map>
#include "graph_constructor.h"
#include <glog/logging.h>

namespace graph{

    class GraphConstructor{
        public:
            GraphConstructor(GraphDef&& gdef, Graph* g)
                :graph_def_(gdef),
                g_(g),
                is_consumed_(graph_def_.node_size(), false){}
            static bool Construct(GraphDef&& gdef, Graph* g);
            bool TryImport(){
                // TF_RETURN_IF_ERROR(EnsureNoNameCollisions());
                // TF_RETURN_IF_ERROR(ValidateInputMapAndControlDependencies());
                BuildNodeIndex();
                InitFromEdges();

                // NOTE: Convert() invokes `consume_node_def()` on each node in the input
                // graph, so `get_node_def()` is no longer usable once it is called.
                Convert();

                FixupSourceAndSinkEdges();
                SetupInputAndOutputNames();
                return true;
            }
        private:
            void BuildNodeIndex();
            void InitFromEdges();
            // make sure the names of input and output should be the same
            void SetupInputAndOutputNames();
            void Convert();
            void FixupSourceAndSinkEdges();
            void MakeNode(NodeDef&& node_def, Node** node);
            void MakeEdge(Node* src, int output_index, Node* dst, int input_index);

            // Decrement pending count for users of `processed` and add the ones that now
            // have all of their pending inputs satisfied to `ready_`.
            void UpdatePendingCountAndReady(int processed);
            GraphDef graph_def_;
            Graph* g_;
            std::vector<bool> is_consumed_;
            struct NodeInfo{
                explicit NodeInfo(int i) : gdef_index(i), node(nullptr) {}
                // Containers require that we have a default constructor.
                NodeInfo() : NodeInfo(-1) {}
                int gdef_index;
                Node* node;  // nullptr until the NodeDef is converted to a Node.
            };
            std::unordered_map<std::string, NodeInfo> gdef_nodes_;

            // Mapping between index within node_defs_ and the index within node_defs_ of
            // all nodes it outputs to.
            std::vector<std::vector<int>> outputs_;


            // Index of NodeDefs in node_defs_ with all inputs already converted. We use a
            // (sorted) set so nodes are created in the order defined in the GraphDef.
            std::set<int> ready_;
            // Mapping between index within node_defs_ and the number of inputs that
            // still need to be converted.
            std::vector<int> pending_count_;

            // Returns the number of nodes in the graph.
            virtual size_t node_def_count() const{ return graph_def_.node().size(); };
            // Returns the i^th node in the graph. Must not be called after
            // consume_node_def(i).
            const NodeDef& get_node_def(int i) const {
                CHECK(!is_consumed_[i])
                    << "NodeDef " << i << " accessed after it was consumed.";
                return graph_def_.node(i);
            }
            // Destructively reads the i^th node in the graph, avoiding a copy if
            // possible. After calling this method, the result of get_node_def(i) is
            // undefined.
            virtual NodeDef consume_node_def(int i){
                CHECK(!is_consumed_[i]) << "NodeDef " << i << " consumed twice.";
                is_consumed_[i] = true;
                return std::move(*graph_def_.mutable_node(i));
            }
    };

    bool GraphConstructor::Construct(GraphDef&& gdef, Graph* g){
        GraphConstructor c(std::move(gdef), g);
        return c.TryImport();
    }

    void GraphConstructor::BuildNodeIndex(){
        // generate gdef_nodes_
        for(int n=0; n<node_def_count(); n++){
            const NodeDef& node_def = get_node_def(n);
            if (!gdef_nodes_
                    .insert(std::make_pair(node_def.name(), NodeInfo(n)))
                    .second) {
                LOG(FATAL)<< "Node '"<< node_def.name()<<"' is not unique";
            }
            // valid op type and device
            if (node_def.type().empty()) {
                LOG(FATAL)<<"Node '"<<node_def.name()
                    <<"' does not specify an operation";
            }
        }
    }

    void GraphConstructor::InitFromEdges(){
        const int num_nodes = node_def_count();
        pending_count_.reserve(num_nodes);
        outputs_.resize(num_nodes);
        for (int n = 0; n < num_nodes; ++n) {
            const NodeDef& node_def = get_node_def(n);
            int pending_count = node_def.input_index_size();
            for (int i = 0; i < node_def.input_index_size(); ++i){
                const int input_index  = node_def.input_index(i);
                auto& input_name = graph_def_.tensor_names(input_index);
                bool is_input=false;

                auto iter = gdef_nodes_.find(input_name);
                if (iter == gdef_nodes_.end()) {
                    LOG(FATAL)<<"Node '"<<node_def.name()
                        <<"': Unknown input node '"
                        <<input_name<<"'";
                }
                outputs_[iter->second.gdef_index].push_back(n);
            }
            if (pending_count == 0) {
                ready_.insert(n);
            }
            pending_count_.push_back(pending_count);
        }
    }

    void GraphConstructor::Convert(){
        struct InputInfo{
            explicit InputInfo(const std::string& node_name, Node* n, int i)
                :name(node_name), node(n), index(i){}
            std::string name;
            Node* node;
            int index;
        };
        std::vector<InputInfo> inputs;
        int processed = 0;
        // bfs
        while(!ready_.empty()){
            // get node from ready_, just process it
            int o = *ready_.begin();
            ready_.erase(ready_.begin());
            ++processed;
            inputs.clear();
            NodeDef node_def = consume_node_def(o);
            // generate inputs
            for (int i = 0; i < node_def.input_index_size(); ++i){
                const int tensor_index = node_def.input_index(i);
                auto& tensor_name = graph_def_.tensor_names(tensor_index);
                // use tensor_name as node name
                auto& node_name = tensor_name;
                Node* src_node;
                int src_index;
                // Locate input in newly-imported nodes
                auto iter = gdef_nodes_.find(node_name);
                DCHECK(iter != gdef_nodes_.end()) << node_name;
                src_node = iter->second.node;
                // only one output tensor in src node
                src_index = 0;
                CHECK(src_node!=nullptr) << "Node '" << node_def.name() << "': Connecting to invalid output "
                    << 0 << " of source node " << node_name
                    << " which has " << src_node->num_outputs() << " outputs.";

                inputs.emplace_back(node_name, src_node, src_index);
            }
            // populate the remain properties
            // const OpDef* op_def;
            // TF_RETURN_IF_ERROR(
            // g_->op_registry()->LookUpOpDef(node_def.op(), &op_def));
            Node* node;

            // final build node
            MakeNode(std::move(node_def), &node);
            // assign Node to NodeInfo
            gdef_nodes_[node->name()].node = node;

            // Add edges from inputs to *node to the graph.
            for (size_t i = 0; i < inputs.size(); ++i){
                MakeEdge(inputs[i].node, inputs[i].index, node, i);
            }
            // infer shape of node
            // ValidateShape(node);

            // Update pending_count_ for outputs.
            UpdatePendingCountAndReady(o);
        }
    }

    void GraphConstructor::FixupSourceAndSinkEdges(){}
    void GraphConstructor::UpdatePendingCountAndReady(int processed) {
        // update ready_ after process the node(the processed one)
        for (size_t i = 0; i < outputs_[processed].size(); ++i) {
            const int output = outputs_[processed][i];
            int* current_pending_count = &pending_count_[output];
            CHECK_GT(*current_pending_count, 0);
            (*current_pending_count)--;
            if (*current_pending_count == 0) {
                ready_.insert(output);
            }
        }
    }

    bool ConvertGraphDefToGraph(const GraphDef& gdef, Graph* g) {
        return ConvertGraphDefToGraph(GraphDef(gdef), g);
    }

    void GraphConstructor::MakeNode(NodeDef&& node_def, Node** node){
        // Add the node to the graph.
        *node = g_->AddNode(std::move(node_def));
    }

    void GraphConstructor::MakeEdge(Node* src, int output_index, Node* dst,
            int input_index) {
        // VLOG(1)<<src->name()<<":"<<output_index<<" -> "
        // <<dst->name()<<":"<<input_index;
        g_->AddEdge(src, output_index, dst, input_index);
    }


    void GraphConstructor::SetupInputAndOutputNames(){
        g_->SetupInputAndOutputNames(graph_def_);
    }


    bool ConvertGraphDefToGraph(GraphDef&& gdef, Graph* g){
        return GraphConstructor::Construct(std::move(gdef), g);
    }
}

