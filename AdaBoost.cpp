#include <cmath>
#include "AdaBoost.h"
#include <iostream>
using namespace adaboost;

Learning::Learning(unsigned short K, weight_t weight_init) : K_{K},
                                                             weight_init_{weight_init}{
}

model_t Learning::train_model(labeled_data_t & data)
{
    model_t model;
    reference_data(data);
    N_ = data.size();
    Decision_stump_learning ds(training_data_);

    init_weights();
    for (unsigned short i = 0; i < K_; ++i){
        auto desicionStump = ds.learn_stump();
        /*compute error*/
        auto error=compute_error(desicionStump);
        /*voting weight*/
        desicionStump.voting_weight=0.5f*log((1-error)/error);
        update_weights(desicionStump);
        model.emplace_back(desicionStump);
    }
    return model;
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

void Learning::update_weights(const DecisionStump & decisionStump){
    Decision_stump_prediction<const WLSample_t> predictor(decisionStump);
    auto Z=0.f;/*Normalizer*/
    for(auto & sample : training_data_) {
        auto prediction=predictor.classify(sample);
        auto f = static_cast<int>(prediction)* static_cast<int>(sample.label);
        sample.weight*=exp(-decisionStump.voting_weight*f);
        Z+=sample.weight;
    }

    for(auto & sample : training_data_) {
        sample.weight/=Z;
    }
}

float Learning::compute_error(const DecisionStump & decisionStump){
    auto error=0.f;
    for(const auto & sample : training_data_){
        error+=sample.weight*I(sample,decisionStump);
    }
    return error;
}


uint16_t Learning::I(const WLSample_t & sample, const DecisionStump & ds){
    Decision_stump_prediction<const WLSample_t> predictor(ds);
    if (predictor.classify(sample) == sample.label){
        return 1u;
    }
    return 0u;
}


