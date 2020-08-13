#ifndef OPENGL_CORE_METRIC_H_
#define OPENGL_CORE_METRIC_H_
#include "opengl/core/types.h"

namespace opengl{
    namespace metrics{
        void UpdateGraphExecTime(const uint64 running_time_usecs);
        void UpdateGraphBuildTime(const uint64 running_time_usecs);
        void UpdateGraphOptimizationPassTime(const string& pass_name,
                const uint64 running_time_usecs);
    }//namespace metrics
}//namespace opengl

#endif
