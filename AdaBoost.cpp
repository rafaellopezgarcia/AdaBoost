#include "AdaBoost.h"
#include <iostream>
using namespace adaboost;

Learning::Learning(unsigned short K, weight_t weight_init) : K_{K},
                                                             weight_init_{weight_init}{
}

void Learning::train_model(const labeled_data_t & data)
{
    N_ = data.size();
    reference_data(data);
    init_weights();

}

void Learning::reference_data(const labeled_data_t & data){
    for(auto & datum : data){
        training_data.emplace_back(Weighted_labeled_sample(datum));
    }
}

void Learning::init_weights(){
    (this->*iw[weight_init_])();
}

void Learning::init_weights_even(){
    for(auto & datum : training_data){
        datum.weight = 1.f / N_;
    }
}

void Learning::recompute_weights(){
    for(auto & datum : training_data){
        Decision_stump_prediction<Weighted_labeled_sample> dsp;
        float Z = compute_normalizer();
        char error= static_cast<char>(dsp.classify(datum)) * static_cast<char>(datum.label);

    }
}

float Learning::compute_normalizer(){
    float Z;
    return Z;
}
void Learning::helper_function(){
    for(auto & t : training_data) {
        for(auto & f : t.features){
            std::cout << f << " ";
        }
        std::cout << static_cast<int>(t.label) << " ";
        std::cout << t.weight << std::endl;
    }

}
