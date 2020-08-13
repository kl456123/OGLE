#ifndef CONVERTER_CORE_CONVERTER_H_
#define CONVERTER_CORE_CONVERTER_H_
#include <string>
#include "core/config.h"
#include "core/registry.h"
#include "core/optimizer.h"
#include "dlcl.pb.h"


class Converter: public RegistryItemBase{
    public:
        Converter(){};
        virtual void Reset(const ConverterConfig config)=0;
        virtual ~Converter(){};
        // TODO(breakpoint) change to return Status
        virtual void Run()=0;

        void Save(std::string checkpoint_path);
        void Save(){
            Save(converter_config_.dst_model_path);
        }

        std::string DebugString()const;

        const dlxnet::ModelProto& model()const{
            return *model_;
        }
        void Optimize(const Optimizer* optimizer);

    protected:
        ConverterConfig converter_config_;
        dlxnet::ModelProto* model_;
};


// INSTANIZE_REGISTRY(Converter);
#define REGISTER_CLASS_CONVERTER(CLASS)  \
    REGISTER_CLASS(Converter, CLASS)

#endif

