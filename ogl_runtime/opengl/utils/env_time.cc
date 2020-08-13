#include <sys/time.h>
#include <time.h>

#include "opengl/utils/env_time.h"

namespace {

    class PosixEnvTime : public EnvTime {
        public:
            PosixEnvTime() {}

            uint64 NowNanos() const override {
                struct timespec ts;
                clock_gettime(CLOCK_REALTIME, &ts);
                return (static_cast<uint64>(ts.tv_sec) * kSecondsToNanos +
                        static_cast<uint64>(ts.tv_nsec));
            }
    };

}  // namespace

EnvTime* EnvTime::Default() {
    static EnvTime* default_env_time = new PosixEnvTime;
    return default_env_time;
}


