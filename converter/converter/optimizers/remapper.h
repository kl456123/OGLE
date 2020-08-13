#ifndef OPTIMIZER_REMAPPER_H_
#define OPTIMIZER_REMAPPER_H_
#include "core/optimizer.h"

namespace optimizer{
    class Remapper :public OptimizationPass{
        public:
            std::string name() const override { return "remapper"; };
            void Run(graph::Graph* graph) override;
    };

}//namespace optimizer

#endif
