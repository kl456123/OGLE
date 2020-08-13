#include "opengl/utils/env.h"

namespace opengl{
    class PosixEnv : public Env {
        public:
            PosixEnv() {}

            ~PosixEnv() override { LOG(FATAL) << "Env::Default() must not be destroyed"; }

            int32 GetCurrentThreadId() override {
                return 0;
            }
            bool GetCurrentThreadName(string* name) override {
                return false;
            }
    };

    Env* Env::Default() {
        static Env* default_env = new PosixEnv;
        return default_env;
    }
}
