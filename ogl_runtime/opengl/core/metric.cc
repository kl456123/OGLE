#include "opengl/core/metric.h"
#include "opengl/core/lib/monitor/counter.h"

namespace opengl{
    namespace metrics{
        namespace{
            // store runtime intermediate result here

            // loop times of running graph
            auto* graph_runs = monitoring::Counter<0>::New(
                    "/tensorflow/core/graph_runs",
                    "The number of graph executions used to collect "
                    "/tensorflow/core/graph_run_time_usecs");

            // time to run graph
            auto* graph_run_time_usecs = monitoring::Counter<0>::New(
                    "/tensorflow/core/graph_run_time_usecs",
                    "The total time spent on executing graphs in microseconds.");

            auto* build_graph_calls = monitoring::Counter<0>::New(
                    "/tensorflow/core/graph_build_calls",
                    "The number of times TensorFlow has created a new client graph. "
                    "A client graph is a sub-graph of the full graph, induced by a set of "
                    "options, including the requested feeds and fetches. It includes time "
                    "spent optimizing the graph with Grappler, and time spent pruning the "
                    "sub-graph.");

            auto* build_graph_time_usecs = monitoring::Counter<0>::New(
                    "/tensorflow/core/graph_build_time_usecs",
                    "The amount of time TensorFlow has spent creating new client graphs in "
                    "microseconds. "
                    "A client graph is a sub-graph of the full graph, induced by a set of "
                    "options, including the requested feeds and fetches. It includes time "
                    "spent optimizing the graph with Grappler, and time spent pruning the "
                    "sub-graph.");
        }//namespace
        void UpdateGraphExecTime(const uint64 running_time_usecs){
            if (running_time_usecs > 0) {
                graph_runs->GetCell()->IncrementBy(1);
                graph_run_time_usecs->GetCell()->IncrementBy(running_time_usecs);
            }
        }
        void UpdateGraphBuildTime(const uint64 running_time_usecs){
            if (running_time_usecs > 0) {
                build_graph_calls->GetCell()->IncrementBy(1);
                build_graph_time_usecs->GetCell()->IncrementBy(running_time_usecs);
            }
        }
        void UpdateGraphOptimizationPassTime(const string& pass_name,
                const uint64 running_time_usecs){
        }
    }//namespace metrics
}//namespace opengl
