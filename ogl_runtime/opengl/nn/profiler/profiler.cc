#include <sstream>
#include <map>
#include "opengl/utils/logging.h"

#include "opengl/nn/profiler/profiler.h"
#include "opengl/core/step_stats.pb.h"

namespace opengl{

    void Profiler::PrintProfiling(const StepStats& step_stats,
            const int step_size, bool verbose){
        // process each node time according to its kernel type
        // name: nsec, count
        NamedStats type2time;
        for(auto& dev_stats:step_stats.dev_stats()){
            VLOG(1)<<"device name: "<<dev_stats.device();
            // loop all nodes which in that device
            for(auto& node_stats: dev_stats.node_stats()){
                const float t = node_stats.all_end_rel_micros()*1e-3;
                string key_name;
                if(verbose||node_stats.node_type().empty()){
                    key_name = node_stats.node_name();
                }else{
                    key_name = node_stats.node_type();
                }
                if(type2time.find(key_name)!=type2time.end()){
                    type2time[key_name].first+=t;
                    type2time[key_name].second++;
                }else{
                    type2time[key_name].first = t;
                    type2time[key_name].second = 1;
                }
            }
        }

        // print time map
        std::stringstream ss;
        ss<<"Node Type\tTime\tCount\n";
        for(auto& iter: type2time){
            ss<<iter.first<<"\t\t"
                <<iter.second.first/step_size<<"ms\t\t"
                <<iter.second.second/step_size<<"\n";
        }

        std::cout<<ss.str()<<std::endl;
    }
}
