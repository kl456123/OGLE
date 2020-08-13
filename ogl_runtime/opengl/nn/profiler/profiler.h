#ifndef OPENGL_NN_PROFILER_PROFILER_H_
#define OPENGL_NN_PROFILER_PROFILER_H_
#include <vector>
#include <map>

#include "opengl/core/types.h"

namespace opengl{
    class StepStats;
    class Profiler{
        public:
            typedef std::map<string, std::pair<float, int>> NamedStats;
            void PrintProfiling(const StepStats& step_stats, const int num_iters,
                    bool verbose=true);
    };
}


#endif
