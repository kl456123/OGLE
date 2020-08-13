#include "opengl/core/lib/monitor/collection_registry.h"
#include "opengl/utils/env.h"

namespace opengl{
    namespace monitoring{
        namespace internal {
            void Collector::CollectMetricValues(
                    const CollectionRegistry::CollectionInfo& info) {
                info.collection_function(MetricCollectorGetter(
                            this, info.metric_def, info.registration_time_millis));
            }

            std::unique_ptr<CollectedMetrics> Collector::ConsumeCollectedMetrics() {
                return std::move(collected_metrics_);
            }

            void Collector::CollectMetricDescriptor(
                    const AbstractMetricDef* const metric_def) {
                auto* const metric_descriptor = [&]() {
                    return collected_metrics_->metric_descriptor_map
                        .insert(std::make_pair(
                                    string(metric_def->name()),
                                    std::unique_ptr<MetricDescriptor>(new MetricDescriptor())))
                        .first->second.get();
                }();
                metric_descriptor->name = string(metric_def->name());
                metric_descriptor->description = string(metric_def->description());

                for (const auto label_name : metric_def->label_descriptions()) {
                    metric_descriptor->label_names.emplace_back(label_name);
                }

                metric_descriptor->metric_kind = metric_def->kind();
                metric_descriptor->value_type = metric_def->value_type();
            }

        }  // namespace internal
        // static
        CollectionRegistry* CollectionRegistry::Default() {
            static CollectionRegistry* default_registry =
                new CollectionRegistry(Env::Default());
            return default_registry;
        }

        CollectionRegistry::CollectionRegistry(Env* const env) : env_(env) {}

        std::unique_ptr<CollectionRegistry::RegistrationHandle>
            CollectionRegistry::Register(const AbstractMetricDef* const metric_def,
                    const CollectionFunction& collection_function) {
                CHECK(collection_function)
                    << "Requires collection_function to contain an implementation.";


                const auto found_it = registry_.find(metric_def->name());
                if (found_it != registry_.end()) {
                    LOG(ERROR) << "Cannot register 2 metrics with the same name: "
                        << metric_def->name();
                    return nullptr;
                }
                registry_.insert(
                        {metric_def->name(),
                        {metric_def, collection_function, env_->NowMicros() / 1000}});

                return std::unique_ptr<RegistrationHandle>(
                        new RegistrationHandle(this, metric_def));
            }

        void CollectionRegistry::Unregister(const AbstractMetricDef* const metric_def) {
            registry_.erase(metric_def->name());
        }

        std::unique_ptr<CollectedMetrics> CollectionRegistry::CollectMetrics(
                const CollectMetricsOptions& options) const {
            internal::Collector collector(env_->NowMicros() / 1000);

            for (const auto& registration : registry_) {
                if (options.collect_metric_descriptors) {
                    collector.CollectMetricDescriptor(registration.second.metric_def);
                }

                collector.CollectMetricValues(registration.second /* collection_info */);
            }
            return collector.ConsumeCollectedMetrics();
        }
    }// namespace monitoring
}//namespace opengl
