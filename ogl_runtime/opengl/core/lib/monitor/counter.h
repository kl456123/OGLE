#ifndef OPENGL_CORE_LIB_MONITOR_COUNTER_H_
#define OPENGL_CORE_LIB_MONITOR_COUNTER_H_
#include "opengl/core/lib/monitor/metric_def.h"
#include "opengl/core/lib/monitor/collection_registry.h"

#include <map>
#include <array>
#include <memory>

namespace opengl{
    namespace monitoring{
        namespace{
            typedef bool Status;
            Status OK=true;
            Status ERROR=false;
        }
        // CounterCell stores each value of an Counter.
        //
        // A cell can be passed off to a module which may repeatedly update it without
        // needing further map-indexing computations. This improves both encapsulation
        // (separate modules can own a cell each, without needing to know about the map
        // to which both cells belong) and performance (since map indexing and
        // associated locking are both avoided).
        //
        // This class is thread-safe.
        class CounterCell {
            public:
                explicit CounterCell(int64 value) : value_(value) {}
                ~CounterCell() {}

                // Atomically increments the value by step.
                // REQUIRES: Step be non-negative.
                void IncrementBy(int64 step);

                // Retrieves the current value.
                int64 value() const;

            private:
                int64 value_;

                DISALLOW_COPY_AND_ASSIGN(CounterCell);
        };

        // A stateful class for updating a cumulative integer metric.
        //
        // This class encapsulates a set of values (or a single value for a label-less
        // metric). Each value is identified by a tuple of labels. The class allows the
        // user to increment each value.
        //
        // Counter allocates storage and maintains a cell for each value. You can
        // retrieve an individual cell using a label-tuple and update it separately.
        // This improves performance since operations related to retrieval, like
        // map-indexing and locking, are avoided.
        //
        // This class is thread-safe.
        template <int NumLabels>
            class Counter {
                public:
                    ~Counter() {
                        // Deleted here, before the metric_def is destroyed.
                        registration_handle_.reset();
                    }
                    // Creates the metric based on the metric-definition arguments.
                    //
                    // Example;
                    // auto* counter_with_label = Counter<1>::New("/tensorflow/counter",
                    //   "Tensorflow counter", "MyLabelName");
                    template <typename... MetricDefArgs>
                        static Counter* New(MetricDefArgs&&... metric_def_args);

                    // Retrieves the cell for the specified labels, creating it on demand if
                    // not already present.
                    template <typename... Labels>
                        CounterCell* GetCell(const Labels&... labels);

                    Status GetStatus() { return status_; }
                private:
                    explicit Counter(
                            const MetricDef<MetricKind::kCumulative, int64, NumLabels>& metric_def)
                        : metric_def_(metric_def),
                        registration_handle_(CollectionRegistry::Default()->Register(
                                    &metric_def_, [&](MetricCollectorGetter getter) {
                                    auto metric_collector = getter.Get(&metric_def_);

                                    for (const auto& cell : cells_) {
                                    metric_collector.CollectValue(cell.first, cell.second.value());
                                    }
                                    })) {
                            if (registration_handle_) {
                                status_ = OK;
                            } else {
                                status_ = ERROR;
                            }
                        }
                    // The metric definition. This will be used to identify the metric when we
                    // register it for collection.
                    const MetricDef<MetricKind::kCumulative, int64, NumLabels> metric_def_;

                    std::unique_ptr<CollectionRegistry::RegistrationHandle> registration_handle_;

                    using LabelArray = std::array<string, NumLabels>;
                    std::map<LabelArray, CounterCell> cells_ ;

                    DISALLOW_COPY_AND_ASSIGN(Counter);
                    Status status_;

            };

        ////
        //  Implementation details follow. API readers may skip.
        ////

        inline void CounterCell::IncrementBy(const int64 step) {
            DCHECK_LE(0, step) << "Must not decrement cumulative metrics.";
            value_ += step;
        }

        inline int64 CounterCell::value() const { return value_; }

        template <int NumLabels>
            template <typename... MetricDefArgs>
            Counter<NumLabels>* Counter<NumLabels>::New(
                    MetricDefArgs&&... metric_def_args) {
                return new Counter<NumLabels>(
                        MetricDef<MetricKind::kCumulative, int64, NumLabels>(
                            std::forward<MetricDefArgs>(metric_def_args)...));
            }

        template <int NumLabels>
            template <typename... Labels>
            CounterCell* Counter<NumLabels>::GetCell(const Labels&... labels){
                // Provides a more informative error message than the one during array
                // construction below.
                static_assert(sizeof...(Labels) == NumLabels,
                        "Mismatch between Counter<NumLabels> and number of labels "
                        "provided in GetCell(...).");

                const LabelArray& label_array = {{labels...}};
                const auto found_it = cells_.find(label_array);
                if (found_it != cells_.end()) {
                    return &(found_it->second);
                }
                return &(cells_
                        .emplace(std::piecewise_construct,
                            std::forward_as_tuple(label_array),
                            std::forward_as_tuple(0))
                        .first->second);
            }
    }//namespace monitoring
}//namespace opengl

#endif
