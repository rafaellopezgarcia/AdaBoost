#pragma once
#include <map>
#include "Decision_stump.h"

namespace adaboost{


    enum class weight_t{
        even
    };



    class Learning{
    public:
        explicit Learning(unsigned short K = 10, weight_t weight_init = weight_t::even);
        void train_model(const labeled_data_t & data);
        void helper_function();

    private:
        // Hyperparameters
        const unsigned short K_; // Number of weak classifiers
        weight_t weight_init_; // Weight initialization type

        // Input
        wlabeled_data_t training_data;
        unsigned int N_; // Number of training samples

        // Output
        classifiers_t model_;

        typedef void(Learning::*fptr)();
        std::map<weight_t, fptr> iw{
            {weight_t::even, &Learning::init_weights_even}
        };

        void reference_data(const labeled_data_t & data);
        void init_weights();
        void init_weights_even();
        void recompute_weights();
        float compute_normalizer();

    };

    template <typename T>
    class Inference{
    public:

        std::vector<label_t> predict_labels(const T & udata);
    private:
        classifiers_t model;
        T udata;

        std::vector<float> compute_confidence();
        std::vector<label_t> classify(std::vector<float> confidence);
    };
}/* Ends adaboost namespace */