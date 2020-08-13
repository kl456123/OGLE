#ifndef CONVERTER_CORE_REGISTRY_H_
#define CONVERTER_CORE_REGISTRY_H_
#include <unordered_map>
#include <string>


// implement registry tools to registry any types of class

class RegistryItemBase;
template<typename T>
class Registry;
template<typename T>
class RegisterHelper;

class RegistryItemBase{
    public:
        RegistryItemBase();
        virtual ~RegistryItemBase();
        virtual void Run()=0;
};

template<typename T>
class Registry{
    public:
        Registry();
        virtual ~Registry();
        // get
        static Registry<T>* Global();

        // insert
        void Register(std::string name, T* data);
        // find
        void LookUp(std::string name, T** data);
    private:
        std::unordered_map<std::string, T*> registry_;
};

// use constructor to auto_register
template<typename T>
class RegisterHelper{
    public:
        RegisterHelper(std::string name, T* data);
};

template<typename T>
RegisterHelper<T>::RegisterHelper(std::string name, T* data){
    Registry<T>::Global()->Register(name, data);
}

// Registry
template<typename T>
Registry<T>::Registry(){
}


template<typename T>
Registry<T>::~Registry(){
    for(const auto&e: registry_){
        delete e.second;
    }
}

template<typename T>
Registry<T>* Registry<T>::Global(){
    static Registry<T>* global = new Registry<T>();
    return global;
}


template<typename T>
void Registry<T>::Register(std::string name, T* data){
    registry_.insert(std::make_pair(name, data));
}

template<typename T>
void Registry<T>::LookUp(std::string name, T** data){
    auto it = registry_.find(name);
    if(it!=registry_.end()){
        *data = it->second;
    }
}


// instanize
#define INSTANIZE_REGISTRY(TYPE)    \
    template class Registry<TYPE>


#define REGISTER_CLASS(REGISTRY_TYPE, CLASS) \
    REGISTER_CLASS_BY_NAME(REGISTRY_TYPE, #CLASS, CLASS)

#define REGISTER_CLASS_BY_NAME(REGISTRY_TYPE, CLASS_NAME, CLASS) \
    REGISTER_CLASS_BY_NAME_UNIQ(__COUNTER__, REGISTRY_TYPE, CLASS_NAME, CLASS)

#define REGISTER_CLASS_BY_NAME_UNIQ(CTR, REGISTRY_TYPE, CLASS_NAME, CLASS)  \
    static RegisterHelper<REGISTRY_TYPE> register_helper_##CTR  =       \
        RegisterHelper<REGISTRY_TYPE>(CLASS_NAME, new CLASS)


// #define REGISTER_CLASS(CLASS)       REGISTER_CLASS_BY_NAME(#CLASS, CLASS)
// #define REGISTER_CLASS_BY_NAME(NAME, CLASS)    REGISTER_CLASS_BY_NAME_UNIQ(__COUNTER__, NAME, CLASS)

// #define REGISTER_CLASS_BY_NAME_UNIQ(ctr, NAME, DATA)           \
// static RegisterHelper register_#ctr  =       \
// new RegisterHelper(NAME, DATA)


#endif
