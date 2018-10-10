#include <algorithm> /*sort, min_element, max_element*/
#include <numeric> /*partial_sum */
#include <cmath> /*fabs*/
#include "Decision_stump.h"

WLSample_t::WLSample_t(Labeled_sample & ls) :
        features(ls.features),
        label(ls.label)
{
}

UWLSample_t::UWLSample_t(const WLSample_t & wls, unsigned short dim) :
        feature(wls.features[dim]),
        label(wls.label),
        weight(wls.weight),
        dim(dim),
        cumsum(0.f)
{
}

UWLSample_t::UWLSample_t()
{
}

std::ostream &operator<<(std::ostream &os, const UWLSample_t &sample) {
    os<<sample.feature<< " ";
    os<< static_cast<int>(sample.label)<< " ";
    os<<sample.weight<< " ";
    os<<sample.dim<<" ";
    os<<sample.cumsum;
    return os;
}

SortingSample_t::SortingSample_t(const std::vector<float> &features, uint32_t ind) :
        features(features),
        ind(ind)
{
}

SortingSample_t &SortingSample_t::operator=(const SortingSample_t & rhs){
    if (this != &rhs){
        ind = rhs.ind;
    }
    return *this;
}


DecisionStump::DecisionStump(unsigned char dimension, float threshold,
                             direction_t direction) :
        dimension(dimension),
        threshold(threshold),
        direction(direction)
{
}

Decision_stump_learning::Decision_stump_learning(WLData_t & training_data):
        training_data_(training_data),
        n_training_samples_(training_data_.size())
{
    if (!training_data.empty()){
        n_dim_ = training_data_[0].features.size();
        order_.resize(n_dim_, std::vector<unsigned int>(n_training_samples_,0));
        sort();
    }
};

DecisionStump Decision_stump_learning::learn_stump(){
    DecisionStump ds;
    auto max=0.f;
    /*select optimal threshold and dimension*/
    std::vector<std::pair<float,float>> training_data_dim(n_training_samples_);
    for(unsigned short dim=0u; dim<n_dim_; ++dim) {
        UWLData_t uwl_data=create_unidimensional_set(dim);
        compute_cum_sum(uwl_data);
        update_optimal_stump(ds, uwl_data, max);
    }
    return ds;
}

void Decision_stump_learning::sort(){
    /* declare and fill sorting_vector */
    std::vector<SortingSample_t> sorting_vector;
    for(int i = 0; i < n_training_samples_; ++i){
        sorting_vector.emplace_back(training_data_[i].features, i);
    }

    /* sort around all dimensions */
    for(unsigned int dim = 0; dim < n_dim_; ++dim) {
        auto sorting_criteria = [&dim](auto l, auto r) {
            return l.features[dim] < r.features[dim];
        };
        std::sort(sorting_vector.begin(), sorting_vector.end(), sorting_criteria);

        /* save order for dim */
        for(unsigned int i = 0; i < sorting_vector.size(); ++i){
            order_[dim][i] = sorting_vector[i].ind;
        }

        /* reindex */
        for(unsigned int i = 0; i < sorting_vector.size(); ++i){
            sorting_vector[i].ind = i;
        }
    }

    /*for(auto it = order_.begin(); it != order_.end(); ++it){
        for (auto it2 = it->begin(); it2 != it->end(); ++it2){
            std::cout << *it2 << std::endl;
        }
        std::cout << std::endl;
    }*/
}

UWLData_t Decision_stump_learning::create_unidimensional_set(uint16_t dim){
    UWLData_t uts;/*unidimensional training_set*/
    auto uws=UWLSample_t(training_data_[order_[dim][0]],dim);
    for(unsigned int i=1; i<n_training_samples_; ++i){
        auto sample=training_data_[order_[dim][i]];
        if(uws.feature==sample.features[dim]){
            if(uws.label==sample.label){
                uws.weight+=sample.weight;
            }
            else if(uws.weight>sample.weight){
                uws.weight-=sample.weight;
            }
            else{
                uws.weight=sample.weight-uws.weight;
                uws.label=sample.label;
            }
        }
        else{
            uts.emplace_back(uws);
            uws=UWLSample_t(sample,dim);
        }
    }
    uts.emplace_back((uws));

    /*for(auto it=uts.begin();it!=uts.end();++it){
        std::cout<<*it<<std::endl;
    }*/
    return uts;
}

void Decision_stump_learning::compute_cum_sum(UWLData_t &uwl_data){
    /*cum sum; threshold at the very left*/
    for (unsigned int i = 0; i < n_training_samples_; ++i){
        const auto & sample = uwl_data[i];
        uwl_data[0].cumsum+=sample.weight* static_cast<int>(sample.label);
    }
    /*cum sum moving threshold to the right*/
    for (unsigned int i=1; i<n_training_samples_; ++i){
        const auto & sample = uwl_data[i-1];
        uwl_data[i].cumsum=sample.cumsum-2*sample.weight* static_cast<int>(sample.label);
    }

    /*for(auto it=uwl_data.begin();it!=uwl_data.end();++it){
        std::cout<<*it<<std::endl;
    }*/
    std::cout<<std::endl;
}

void Decision_stump_learning::update_optimal_stump(DecisionStump &ds,UWLData_t &uwl_data,
                                                   float &max_cumsum){
    auto criteria=[](auto &a, auto&b){return a.cumsum<b.cumsum;};

    auto max=std::max_element(uwl_data.begin(),uwl_data.end(), criteria);
    auto min=std::min_element(uwl_data.begin(),uwl_data.end(), criteria);
    auto virtual_optimal=fabs(max->cumsum)>fabs(min->cumsum) ? max : min;

    if(fabs(virtual_optimal->cumsum)>fabs(max_cumsum)){
        max_cumsum=virtual_optimal->cumsum;
        ds.threshold=virtual_optimal->feature;
        ds.direction=max_cumsum>0 ? direction_t::left : direction_t::right;
        //std::cout<<"updated "<<max_cumsum<<std::endl;
    }
}




