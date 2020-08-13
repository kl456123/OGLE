#ifndef OPENGL_NN_PROFILER_PROFILER_INTERFACE_H_
#define OPENGL_NN_PROFILER_PROFILER_INTERFACE_H_
#include <memory>
#include <vector>

#include "opengl/utils/status.h"

namespace opengl{
    class StepStats;
    namespace profiler {

        enum class DeviceType {
            kUnspecified,
            kCpu,
            kGpu,
            kTpu,
        };

        struct ProfilerOptions {
            // DeviceType::kUnspecified: All registered device profiler will be enabled.
            // DeviceType::kCpu: only CPU will be profiled.
            // DeviceType::kGpu: only CPU/GPU will be profiled.
            // DeviceType::kTpu: only CPU/TPU will be profiled.
            DeviceType device_type = DeviceType::kUnspecified;

            // Inexpensive ops are not traced by default.
            int host_tracer_level = 2;
        };

        // Interface for tensorflow profiler plugins.
        //
        // ProfileSession calls each of these methods at most once per instance, and
        // implementations can rely on that guarantee for simplicity.
        //
        // Thread-safety: Implementations are only required to be go/thread-compatible.
        // ProfileSession is go/thread-safe and synchronizes access to ProfilerInterface
        // instances.
        class ProfilerInterface {
            public:
                virtual ~ProfilerInterface() = default;

                // Starts profiling.
                virtual Status Start() = 0;

                // Stops profiling.
                virtual Status Stop() = 0;

                // Saves collected profile data into step_stats_collector.
                // After this or the overload below are called once, subsequent calls might
                // return empty data.
                virtual Status CollectData(StepStats* step_stats) = 0;

                // Which device this ProfilerInterface is used for.
                virtual DeviceType GetDeviceType() = 0;
        };
    }//namespace profiler

    using ProfilerFactory = std::unique_ptr<profiler::ProfilerInterface> (*)(
            const profiler::ProfilerOptions&);

    void RegisterProfilerFactory(ProfilerFactory factory);

    void CreateProfilers(
            const profiler::ProfilerOptions& options,
            std::vector<std::unique_ptr<profiler::ProfilerInterface>>* result);
}//namespace opengl


#endif
