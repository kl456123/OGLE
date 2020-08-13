#include "opengl/nn/profiler/profiler_interface.h"

namespace opengl{
    namespace {
        std::vector<ProfilerFactory>* GetFactories() {
            static auto factories = new std::vector<ProfilerFactory>();
            return factories;
        }
    }  // namespace

    void RegisterProfilerFactory(ProfilerFactory factory) {
        GetFactories()->push_back(factory);
    }

    void CreateProfilers(
            const profiler::ProfilerOptions& options,
            std::vector<std::unique_ptr<profiler::ProfilerInterface>>* result) {
        for (auto factory : *GetFactories()) {
            if (auto profiler = factory(options)) {
                result->push_back(std::move(profiler));
            }
        }
    }
}//namespace opengl
