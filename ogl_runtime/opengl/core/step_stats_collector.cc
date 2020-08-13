#include "opengl/core/step_stats_collector.h"
#include "opengl/core/step_stats.pb.h"
#include "opengl/core/kernel.h"
#include "opengl/core/tensor.h"
#include "opengl/utils/logging.h"
#include "opengl/utils/env.h"
#include "opengl/utils/strings.h"

namespace opengl{
    NodeExecStatsWrapper::NodeExecStatsWrapper(
            const Kernel* node, StepStatsCollector* step_stats_collector)
        : NodeExecStatsWrapper(std::unique_ptr<NodeExecStats>(new NodeExecStats), node,
                step_stats_collector) {
            stats_->set_node_name(node->kernel_name());
            stats_->set_node_type(node->kernel_type());
        }

    NodeExecStatsWrapper::NodeExecStatsWrapper(
            std::unique_ptr<NodeExecStats> stats, const Kernel* node,
            StepStatsCollector* step_stats_collector)
        : stats_(std::move(stats)),
        node_(node),
        step_stats_collector_(step_stats_collector) {}

    void NodeExecStatsWrapper::Done(const string& device) {
        DCHECK(node_);
        string memory;

        auto text = strings::StrCat(memory, node_->kernel_name(), " = ", node_->kernel_type());
        stats_->set_timeline_label(text);
        step_stats_collector_->Save(device, this);
    }

    void NodeExecStatsWrapper::RecordExecutorStarted() {
        int64 now_nanos = Env::Default()->NowNanos();
        stats_->set_all_start_micros(now_nanos / EnvTime::kMicrosToNanos);
        stats_->set_all_start_nanos(now_nanos);
    }



    void NodeExecStatsWrapper::RecordComputeStarted() {
        int64 now_nanos = Env::Default()->NowNanos();
        DCHECK_NE(stats_->all_start_micros(), 0);
        DCHECK_NE(stats_->all_start_nanos(), 0);
        stats_->set_op_start_rel_micros(now_nanos / EnvTime::kMicrosToNanos -
                stats_->all_start_micros());
        stats_->set_op_start_rel_nanos(now_nanos - stats_->all_start_nanos());
    }

    void NodeExecStatsWrapper::RecordComputeEnded() {
        int64 now_nanos = Env::Default()->NowNanos();
        DCHECK_NE(stats_->all_start_micros(), 0);
        DCHECK_NE(stats_->all_start_nanos(), 0);
        stats_->set_op_end_rel_micros(now_nanos / EnvTime::kMicrosToNanos -
                stats_->all_start_micros());
        stats_->set_op_end_rel_nanos(now_nanos - stats_->all_start_nanos());
    }

    void NodeExecStatsWrapper::RecordExecutorEnded() {
        int64 now_nanos = Env::Default()->NowNanos();
        DCHECK_NE(stats_->all_start_micros(), 0);
        DCHECK_NE(stats_->all_start_nanos(), 0);
        stats_->set_all_end_rel_micros(now_nanos / EnvTime::kMicrosToNanos -
                stats_->all_start_micros());
        stats_->set_all_end_rel_nanos(now_nanos - stats_->all_start_nanos());
    }

    void NodeExecStatsWrapper::SetScheduled(int64 nanos) {
        stats_->set_scheduled_micros(nanos / EnvTime::kMicrosToNanos);
        stats_->set_scheduled_nanos(nanos);
    }


    void NodeExecStatsWrapper::SetOutput(int slot, const Tensor* tensor) {
        DCHECK(tensor);
        NodeOutput* node_output = stats_->add_output();
        node_output->set_slot(slot);
        tensor->FillDescription(node_output->mutable_tensor_description());
    }

    // void NodeExecStatsWrapper::AddAllocation(
    // Allocator* allocator, TrackingAllocator* tracking_allocator) {
    // AllocatorMemoryUsed* memory = stats_->add_memory();
    // memory->set_allocator_name(allocator->Name());
    // auto sizes = tracking_allocator->GetSizes();
    // memory->set_total_bytes(std::get<0>(sizes));
    // memory->set_peak_bytes(std::get<1>(sizes));
    // memory->set_live_bytes(std::get<2>(sizes));

    // absl::optional<AllocatorStats> stats = allocator->GetStats();
    // if (stats) {
    // memory->set_allocator_bytes_in_use(stats->bytes_in_use);
    // }
    // allocations_.push_back(std::make_pair(memory, tracking_allocator));
    // }

    void NodeExecStatsWrapper::Finalize() {
        // for (auto& alloc : allocations_) {
        // AllocatorMemoryUsed* memory = alloc.first;
        // for (auto& record : alloc.second->GetRecordsAndUnRef()) {
        // auto* r = memory->add_allocation_records();
        // r->set_alloc_bytes(record.alloc_bytes);
        // r->set_alloc_micros(record.alloc_micros);
        // }
        // }
        // allocations_.clear();
    }

    StepStatsCollector::StepStatsCollector(StepStats* step_stats)
        : finalized_(false), step_stats_(step_stats) {}

    void StepStatsCollector::Save(const string& device,
            NodeExecStats* node_stats_pb) {
        Save(device,
                new NodeExecStatsWrapper(std::unique_ptr<NodeExecStats>(node_stats_pb),
                    nullptr, this));
    }

    void StepStatsCollector::Save(const string& device,
            NodeExecStatsWrapper* node_stats) {
        if (!node_stats) return;
        VLOG(1) << "Save dev " << device << " node stats " << node_stats->stats();
        {
            if (finalized_) {
                LOG(WARNING) << "stats saved after finalize will not be collected.";
            }
            if (!step_stats_ || collected_nodes_ >= kMaxCollectedNodes) {
                VLOG(1) << "step_stats_ nullptr or already collected too many nodes.";
                delete node_stats;
                return;
            }
            auto& device_stats = dev_stats_[device];
            device_stats.push_back(std::unique_ptr<NodeExecStatsWrapper>(node_stats));
            collected_nodes_++;
        }
    }

    NodeExecStatsInterface* StepStatsCollector::CreateNodeExecStats(
            const Kernel* node) {
        // Only collect statistics for non-transfer nodes.
        return new NodeExecStatsWrapper(node, this);
    }

    string StepStatsCollector::ReportAllocsOnResourceExhausted(const string& err) {
        if (err.find("OOM") == err.npos) {
            return "";
        }
    }

    void StepStatsCollector::Finalize() {
        FinalizeInternal();
    }

    void StepStatsCollector::FinalizeAndSwap(StepStats* step_stats) {
        CHECK(step_stats_);
        FinalizeInternal();
        step_stats->Swap(step_stats_);
        collected_nodes_ = 0;
    }

    void StepStatsCollector::FinalizeInternal() {
        if (!step_stats_ || finalized_) {
            return;
        }
        finalized_ = true;
        std::map<string, DeviceStepStats*> dev_stats_pb;
        for (auto& ds : *step_stats_->mutable_dev_stats()) {
            dev_stats_pb[ds.device()] = &ds;
        }
        for (const auto& dev_stat : dev_stats_) {
            if (dev_stats_pb.find(dev_stat.first) == dev_stats_pb.end()) {
                DeviceStepStats* ndev_stat = step_stats_->add_dev_stats();
                ndev_stat->set_device(dev_stat.first);
                dev_stats_pb[dev_stat.first] = ndev_stat;
            }
            DeviceStepStats* dss = dev_stats_pb.at(dev_stat.first);
            for (auto& stats : dev_stat.second) {
                stats->Finalize();
                stats->stats()->Swap(dss->add_node_stats());
            }
        }

        for (const auto& device_thread : thread_names_) {
            if (dev_stats_pb.find(device_thread.first) == dev_stats_pb.end()) {
                // skip device without DeviceStepStats.
                continue;
            }
            DeviceStepStats* dss = dev_stats_pb.at(device_thread.first);
            for (const auto& thread_name : device_thread.second) {
                (*dss->mutable_thread_names())[thread_name.first] = thread_name.second;
            }
        }
    }

    void StepStatsCollector::SaveThreadName(const string& device,
            const uint32 thread_id,
            const string& thread_name) {
        VLOG(1) << "Save dev " << device << " thread id " << thread_id << " name "
            << thread_name;
        {
            if (finalized_) {
                LOG(WARNING) << "thread_name saved after finalize will not be collected.";
            }
            auto& thread_names_map = thread_names_[device];
            thread_names_map[thread_id] = thread_name;
        }
    }
}//namespace opengl
