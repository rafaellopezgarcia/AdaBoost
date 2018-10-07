#include "AdaBoost.h"
#include <iostream>
using namespace adaboost;


Learning::Learning(unsigned short K, weight_t weight_init) : K_{K},
                                                             weight_init_{weight_init}{
}

void Learning::train_model(labeled_data_t & data)
{
    reference_data(data);
    Decision_stump_learning ds(training_data_);
    N_ = data.size();

    init_weights();
    for (unsigned short i = 0; i < K_; ++i){
        model_.emplace_back(ds.learn_stump());
    }

}

void Learning::reference_data(labeled_data_t & data){
    for(auto & datum : data){
        training_data_.emplace_back(WLSample_t(datum));
    }
}

void Learning::init_weights(){
    (this->*iw[weight_init_])();
}

void Learning::init_weights_even(){
    for(auto & datum : training_data_){
        datum.weight = 1.f / N_;
    }
}

void Learning::recompute_weights(){
    for(auto & datum : training_data_){
        //Decision_stump_prediction<WLSample_t> dsp;
        float Z = compute_normalizer();
        //char error= static_cast<char>(dsp.classify(datum)) * static_cast<char>(datum.label);

    }
}

float Learning::compute_normalizer(){
    float Z;
    return Z;
}
void Learning::helper_function(){
    /*for(auto & t : training_data_) {
        for(auto & f : t.features){
            std::cout << f << " ";
        }
        std::cout << static_cast<int>(t.label) << " ";
        std::cout << t.weight << std::endl;
    }*/

}
