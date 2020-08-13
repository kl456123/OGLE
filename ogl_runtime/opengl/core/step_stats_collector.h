#ifndef OPENGL_CORE_STEP_STATS_COLLECTOR_H_
#define OPENGL_CORE_STEP_STATS_COLLECTOR_H_
#include <memory>
#include <unordered_map>
#include <vector>
#include "opengl/core/types.h"

namespace opengl{
    class Kernel;
    class StepStats;
    class StepStatsCollector;
    class NodeExecStats;
    // Statistics collection interface for individual node execution.
    //
    // See `NodeExecStatsWrapper` for a concrete implementation of this interface
    // that interfaces with the `Session` layer.
    class NodeExecStatsInterface {
        public:
            virtual ~NodeExecStatsInterface() {}

            // Called when the statistics collection for the node has finished. Once this
            // method is called, the caller should not make assumptions about the validity
            // of this object.
            virtual void Done(const string& device) = 0;

            // Called immediately after this node starts being processed by the executor.
            virtual void RecordExecutorStarted() = 0;

            // Called immediately before this node's `Compute()` or `ComputeAsync()`
            // method is called.
            virtual void RecordComputeStarted() = 0;

            // Called immediately after this node's `Compute()` method returned (or, for
            // asynchronous operations, the callback passed to its `ComputeAsync()` method
            // was called).
            virtual void RecordComputeEnded() = 0;

            // Called immediately after this executor finishes processing this node.
            virtual void RecordExecutorEnded() = 0;

            // Returns `true` if this object should track memory allocations.
            virtual bool TrackAllocations() const = 0;

            // Records information about the memory allocated during the execution of this
            // node.
            //
            // Takes ownership of any `TrackingAllocator` objects stored in `ctx`.
            // virtual void SetMemory(OpKernelContext* ctx) = 0;

            // Records information about the tensor produced by this node at the given
            // output slot.
            virtual void SetOutput(int slot, const Tensor* tensor) = 0;

            // Records the absolute time in nanoseconds at which this node became
            // runnable (i.e. was scheduled for execution).
            virtual void SetScheduled(int64 nanos) = 0;
    };

    // Wraps NodeExecStats and adds allocation to it.
    class NodeExecStatsWrapper : public NodeExecStatsInterface {
        public:
            // Does not take ownership of `node` or `step_stats_collector`.
            NodeExecStatsWrapper(const Kernel* node,
                    StepStatsCollector* step_stats_collector);

            // Takes ownership of 'stats' but not `node` or `step_stats_collector`.
            NodeExecStatsWrapper(std::unique_ptr<NodeExecStats> stats,
                    const Kernel* node,
                    StepStatsCollector* step_stats_collector);

            // Destructor calls Finalize() to release the TrackingAllocators.
            ~NodeExecStatsWrapper() override { Finalize(); }
            void Done(const string& device) override;
            void RecordExecutorStarted() override;
            void RecordComputeStarted() override;
            void RecordComputeEnded() override;
            void RecordExecutorEnded() override;
            bool TrackAllocations() const override { return true; }
            void SetOutput(int slot, const Tensor* tensor) override;
            void SetScheduled(int64 nanos) override;

        private:
            friend class StepStatsCollector;

            NodeExecStats* stats() { return stats_.get(); }

            // Populates stats_ and releases TrackingAllocator.
            void Finalize();

            // Does not take ownership of the `allocator`.
            // Takes ownership of `tracking_allocator`.
            // void AddAllocation(Allocator* allocator,
            // TrackingAllocator* tracking_allocator);

            std::unique_ptr<NodeExecStats> stats_;
            const Kernel* const node_;                       // Not owned.
            StepStatsCollector* const step_stats_collector_;  // Not owned.
    };

    // Statistics collection interface for step execution.
    //
    // See `StepStatsCollector` for a concrete implementation of this interface
    // that interfaces with the `Session` layer.
    class StepStatsCollectorInterface {
        public:
            virtual ~StepStatsCollectorInterface() {}

            // Creates an instance of `NodeExecStatsInterface` that should be used for
            // collecting statistics about individual node execution.
            virtual NodeExecStatsInterface* CreateNodeExecStats(const Kernel* node) = 0;

            // Generates a string reporting the currently used memory based
            // on ResourceExhausted OOM `err` message.
            // `err` message needs to contain device name and allocator name, e.g.:
            // "ResourceExhaustedError: OOM when allocating tensor ...
            // on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc"
            virtual string ReportAllocsOnResourceExhausted(const string& err) = 0;
    };

    // StepStatsCollector manages the collection of a StepStats object.
    // The StepStats object holds multiple DeviceStats.
    // Each DeviceStats object holds multiple NodeExecStats.
    class StepStatsCollector : public StepStatsCollectorInterface {
        public:
            // Does not take ownership of `step_stats`.
            explicit StepStatsCollector(StepStats* step_stats);

            // Saves node statistics to the DeviceStats object associated with device.
            // Should be called before Finalize.
            void Save(const string& device, NodeExecStats* node_stats_pb);
            void Save(const string& device, NodeExecStatsWrapper* node_stats);

            // Saves thread name.
            void SaveThreadName(const string& device, const uint32 thread_id,
                    const string& thread_name);

            NodeExecStatsInterface* CreateNodeExecStats(const Kernel* node) override;
            string ReportAllocsOnResourceExhausted(const string& err) override;

            // The following 2 Finalize methods populate the StepStats passed
            // from the constructor. Calling it more than once won't have any effect.
            // User shouldn't call Save() methods after Finalize.
            void Finalize();
            // swaps the content of StepStats* from constructor with 'ss'.
            void FinalizeAndSwap(StepStats* step_stats);

        private:
            // TODO(suharshs): Make this configurable if its not possible to find a value
            // that works for all cases.
            static const uint64 kMaxCollectedNodes = 1 << 20;

            typedef std::vector<std::unique_ptr<NodeExecStatsWrapper>> NodeStatsVector;
            typedef std::unordered_map<uint32, string> ThreadNamesMap;

            void FinalizeInternal();

            bool finalized_;
            std::unordered_map<string, NodeStatsVector> dev_stats_ ;
            std::unordered_map<string, ThreadNamesMap> thread_names_;
            StepStats* step_stats_ ;
            uint64 collected_nodes_ = 0;
    };
}// namespace opengl

#endif
