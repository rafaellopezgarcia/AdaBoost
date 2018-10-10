#pragma once
#include <map>
#include "Decision_stump.h"

namespace adaboost{
    using model_t = std::vector<DecisionStump>;
    enum class weight_t{
        even
    };

    class Learning{
    public:
        explicit Learning(unsigned short K = 10, weight_t weight_init = weight_t::even);
        model_t train_model(labeled_data_t & data);
    private:
        // Hyperparameters
        const unsigned short K_; // Number of weak classifiers
        weight_t weight_init_; // Weight initialization type

        // Input
        WLData_t training_data_;
        unsigned long N_; // Number of training samples

        typedef void(Learning::*fptr)();
        std::map<weight_t, fptr> iw{
            {weight_t::even, &Learning::init_weights_even}
        };

        void reference_data(labeled_data_t & data);
        void init_weights();
        void init_weights_even();
        void update_weights(const DecisionStump & decisionStump);
        float compute_error(const DecisionStump & decisionStump);

        uint16_t I(const WLSample_t & sample, const DecisionStump & ds);
    };

    template <typename T>
    class Inference{
    public:
        std::vector<label_t> predict_labels(const std::vector<T> & samples) {
            std::vector<label_t> predictions;
            for(const auto & sample : samples){
                predict_label(sample);
            }
        }

        label_t predict_label(const T & sample){
            auto confidence = 0.f;
            for(const auto & ds : model_){
                Decision_stump_prediction<T> dsp(ds);
                confidence+=ds.voting_weight*static_cast<uint16_t >(dsp.classify(ds));
            }
            if (confidence<=0) {
                return label_t::class0;
            }
            else{
                return label_t::class1;
            }
        }
    private:
        model_t model_;
    };
}/* Ends adaboost namespace */