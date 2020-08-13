#ifndef CONVERTER_GRAPH_GRAPH_H_
#define CONVERTER_GRAPH_GRAPH_H_
/* use graph class to do some template deduction like op fusion, op conversion,
 * op decomposition
 */

// #include "graph/node.h"
// #include "graph/edge.h"
#include <string>
#include <vector>
#include <unordered_set>
#include "dlcl.pb.h"

namespace graph{
    class Graph;
    class Node;
    class Edge;
    struct NodeProperties;

    class Node{
        public:
            std::string DebugString() const;
            int id() const { return id_; }
            const std::string& name() const;
            void set_name(std::string name);
            const std::string& type_string() const;
            const ::dlxnet::NodeProto& def() const;
            ::dlxnet::NodeProto& def();

            // input and output
            int32_t num_outputs() const;
            int32_t num_inputs() const;
            void input_edge(int idx, const Edge** e) const;
            void output_edge(int idx, const Edge** e)const;
            // more helpful funcs
            const Edge* input_edge(int idx)const;
            const Edge* output_edge(int idx)const;

            const std::set<const Edge*>& in_edges() const { return in_edges_; }
            const std::set<const Edge*>& out_edges() const { return out_edges_; }
            void Clear();
        private:
            enum NodeClass{
                NC_UNINITIALIZED,
                NC_OTHER
            };
            friend class Graph;
            Node();
            Graph* graph_;
            int id_;       // -1 until Initialize() is called
            NodeClass class_;
            NodeProperties* properties() const { return props_.get(); }

            void Initialize(int id,  std::shared_ptr<NodeProperties> props);

            std::set<const Edge*> in_edges_;
            std::set<const Edge*> out_edges_;
            std::shared_ptr<NodeProperties> props_;

    };

    class Edge{
        public:
            Node* src() const { return src_; }
            Node* dst() const { return dst_; }
            int id() const { return id_; }

            // Return the index of the source output that produces the data
            // carried by this edge.  The special value kControlSlot is used
            // for control dependencies.
            int src_output() const { return src_output_; }

            // Return the index of the destination input that consumes the data
            // carried by this edge.  The special value kControlSlot is used
            // for control dependencies.
            int dst_input() const { return dst_input_; }
            std::string DebugString() const;

        private:
            Edge() {}

            friend class Graph;
            Node* src_;
            Node* dst_;
            int id_;
            int src_output_;
            int dst_input_;
    };

    class Graph{
        public:
            explicit Graph();
            virtual ~Graph();

            // Adds a new node to this graph, and returns it. Infers the Op and
            // input/output types for the node. *this owns the returned instance.
            // Returns nullptr and sets *status on error.
            Node* AddNode(::dlxnet::NodeProto node_def);

            // Removes a node from this graph, including all edges from or to it.
            // *node should not be accessed after calling this function.
            // REQUIRES: node->IsOp()
            void RemoveNode(Node* node);
            // Adds an edge that connects the xth output of `source` to the yth input of
            // `dest` and returns it. Does not update dest's NodeDef.
            const Edge* AddEdge(Node* source, int x, Node* dest, int y);

            // Removes edge from the graph. Does not update the destination node's
            // NodeDef.
            // REQUIRES: The edge must exist.
            void RemoveEdge(const Edge* edge);


            // Updates the input to a node.  The existing edge to `dst` is removed and an
            // edge from `new_src` to `dst` is created. The NodeDef associated with `dst`
            // is also updated.
            bool UpdateEdge(Node* new_src, int new_src_index, Node* dst, int dst_index);

            // Because edges can be removed from the graph, num_edges() is often
            // smaller than num_edge_ids(). If one needs to create an array of
            // edges indexed by edge ids, num_edge_ids() should be used as the
            // array's size.
            int num_edges() const { return num_edges_; }


            // Generate new node name with the specified prefix that is unique
            // across this graph.
            std::string NewName(std::string prefix);


            // Returns one more than the maximum id assigned to any node.
            int num_node_ids() const { return nodes_.size(); }

            // Returns the node associated with an id, or nullptr if no node
            // with that id (the node with that id was removed and the id has
            // not yet been re-used). *this owns the returned instance.
            // REQUIRES: 0 <= id < num_node_ids().
            Node* FindNodeId(int id) const { return nodes_[id]; }

            // Returns one more than the maximum id assigned to any edge.
            int num_edge_ids() const { return edges_.size(); }

            // Returns the Edge associated with an id, or nullptr if no edge
            // with that id (the node with that id was removed and the id has
            // not yet been re-used). *this owns the returned instance.
            // REQUIRES: 0 <= id < num_node_ids().
            const Edge* FindEdgeId(int id) const { return edges_[id]; }

            // The number of live nodes in the graph.
            //
            // Because nodes can be removed from the graph, num_nodes() is often
            // smaller than num_node_ids(). If one needs to create an array of
            // nodes indexed by node ids, num_node_ids() should be used as the
            // array's size.
            int num_nodes() const { return num_nodes_; }


            void ToGraphDef(::dlxnet::GraphProto* graph_def) const;

            // called by graph constructor
            void SetupInputAndOutputNames(const ::dlxnet::GraphProto& graph_def);
        private:
            // Ownership of the returned Node is not transferred to caller.
            Node* AllocateNode(std::shared_ptr<NodeProperties> props);
            void ReleaseNode(Node* node);
            // Insert edge in free_edges_ for possible reuse.
            void RecycleEdge(const Edge* edge);

            // Map from node ids to allocated nodes.  nodes_[id] may be nullptr if
            // the node with that id was removed from the graph.
            std::vector<Node*> nodes_;

            // Number of nodes alive.
            int64_t num_nodes_ = 0;

            // Map from edge ids to allocated edges.  edges_[id] may be nullptr if
            // the edge with that id was removed from the graph.
            std::vector<Edge*> edges_;

            // The number of entries in edges_ that are not nullptr.
            int num_edges_ = 0;

            // For generating unique names.
            int name_counter_ = 0;

            // input names and output names
            std::vector<std::string> input_names_;
            std::vector<std::string> output_names_;

    };
}//namespace

#endif
