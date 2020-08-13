#include <memory>

#include "opengl/nn/profiler/traceme_recorder.h"
#include "opengl/nn/profiler/host_tracer_utils.h"
#include "opengl/nn/profiler/profiler_interface.h"
#include "opengl/core/step_stats_collector.h"
#include "opengl/utils/errors.h"
#include "opengl/utils/env_time.h"
#include "opengl/core/step_stats.pb.h"


namespace opengl{
    namespace profiler {
        namespace {
            // Controls TraceMeRecorder and converts TraceMeRecorder::Events into
            // RunMetadata messages.
            //
            // Thread-safety: This class is go/thread-compatible.
            class HostTracer : public ProfilerInterface {
                public:
                    explicit HostTracer(int host_trace_level);
                    ~HostTracer() override;

                    // Starts recording TraceMes.
                    Status Start() override;

                    // Stops recording TraceMes.
                    Status Stop() override;

                    // Populates user traces and thread names in response.
                    // The user traces and thread names are in no particular order.
                    Status CollectData(StepStats* step_stats) override;

                    DeviceType GetDeviceType() override { return DeviceType::kCpu; }

                private:
                    // Level of host tracing.
                    const int host_trace_level_;

                    // True if currently recording.
                    bool recording_ = false;

                    // Timestamp at the start of tracing.
                    uint64 start_timestamp_ns_ = 0;

                    // Container of all traced events.
                    TraceMeRecorder::Events events_;
            };

            HostTracer::HostTracer(int host_trace_level)
                : host_trace_level_(host_trace_level) {}

            HostTracer::~HostTracer() { Stop().IgnoreError(); }

            Status HostTracer::Start() {
                if (recording_) {
                    return Status(::dlxnet::error::INTERNAL, "TraceMeRecorder already started");
                }
                recording_ = TraceMeRecorder::Start(host_trace_level_);
                if (!recording_) {
                    return Status(::dlxnet::error::INTERNAL, "Failed to start TraceMeRecorder");
                }
                start_timestamp_ns_ = EnvTime::Default()->NowNanos();
                return Status::OK();
            }

            Status HostTracer::Stop() {
                if (!recording_) {
                    return Status(dlxnet::error::INTERNAL, "TraceMeRecorder not started");
                }
                events_ = TraceMeRecorder::Stop();
                recording_ = false;
                return Status::OK();
            }

            Status HostTracer::CollectData(StepStats* step_stats) {
                if (recording_) {
                    return errors::Internal("TraceMeRecorder not stopped");
                }
                MakeCompleteEvents(&events_);
                StepStatsCollector step_stats_collector(step_stats);

                constexpr char kUserMetadataMarker = '#';
                const string cpu_name = "/host:CPU";

                for (auto& thread : events_) {
                    step_stats_collector.SaveThreadName(cpu_name, thread.thread.tid,
                            thread.thread.name);
                    for (auto& event : thread.events) {
                        if (event.start_time && event.end_time) {
                            NodeExecStats* ns = new NodeExecStats;
                            if (event.name.back() != kUserMetadataMarker) {
                                ns->set_node_name(std::move(event.name));
                            } else {
                                // // Expect the format will be "<node_name>#<node_type>#"
                                std::vector<string> parts;
                                strings::split(event.name, parts, "#");
                                if (parts.size() >= 2) {
                                    ns->set_node_name(parts[0]);
                                    ns->set_node_type(parts[1]);
                                } else {
                                    ns->set_node_name(std::move(event.name));
                                }
                            }
                            ns->set_all_start_micros(event.start_time / EnvTime::kMicrosToNanos);
                            ns->set_all_end_rel_micros((event.end_time - event.start_time) /
                                    EnvTime::kMicrosToNanos);
                            ns->set_thread_id(thread.thread.tid);
                            step_stats_collector.Save(cpu_name, ns);
                        }
                    }
                }
                events_.clear();
                step_stats_collector.Finalize();
                return Status::OK();
            }
        } // namespace

        // Not in anonymous namespace for testing purposes.
        std::unique_ptr<ProfilerInterface> CreateHostTracer(
                const profiler::ProfilerOptions& options) {
            if (options.host_tracer_level == 0) return nullptr;
            return std::unique_ptr<HostTracer>(new HostTracer(options.host_tracer_level));
        }

        auto register_host_tracer_factory = [] {
            bool enable=true;

            // TF_CHECK_OK(ReadBoolFromEnvVar("TF_ENABLE_OSS_CPU_PROFILER", true, &enable));
            if (enable) {
                RegisterProfilerFactory(&CreateHostTracer);
            }
            return 0;
        }();
    } // namespace profiler
} // namespace opengl
