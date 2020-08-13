#include "opengl/nn/profiler/profiler_interface.h"
#include "opengl/utils/logging.h"


namespace opengl{
    namespace profiler{
        // GpuTracer for GPU.
        class GpuTracer : public profiler::ProfilerInterface {
            public:
                GpuTracer(){
                    VLOG(1) << "GpuTracer created.";
                }

                ~GpuTracer() override {}

                // GpuTracer interface:
                Status Start() override;
                Status Stop() override;
                Status CollectData(StepStats* step_stats) override;
                profiler::DeviceType GetDeviceType() override {
                    return profiler::DeviceType::kGpu;
                }
        };

        Status GpuTracer::Start() {
            return Status::OK();
        }

        Status GpuTracer::Stop() {
            return Status::OK();
        }
        Status GpuTracer::CollectData(StepStats* step_stats) {
        }
    } // namespace profiler
    // Not in anonymous namespace for testing purposes.
    std::unique_ptr<profiler::ProfilerInterface> CreateGpuTracer(
            const profiler::ProfilerOptions& options) {
        if (options.device_type != profiler::DeviceType::kGpu &&
                options.device_type != profiler::DeviceType::kUnspecified)
            return nullptr;
        // profiler::CuptiTracer* cupti_tracer =
        // profiler::CuptiTracer::GetCuptiTracerSingleton();
        // if (!cupti_tracer->IsAvailable()) {
        // return nullptr;
        // }
        // profiler::CuptiInterface* cupti_interface = profiler::GetCuptiInterface();
        return std::make_unique<profiler::GpuTracer>();
    }

    auto register_gpu_tracer_factory = [] {
        RegisterProfilerFactory(&CreateGpuTracer);
        return 0;
    }();
} // namespace opengl
