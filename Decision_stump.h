#pragma once
#include <vector>
#include "Supervised_learning.h"

struct Weighted_labeled_sample;
struct DecisionStump;
using wlabeled_data_t = std::vector<Weighted_labeled_sample>;
using classifiers_t = std::vector<DecisionStump>;

enum class direction_t{
    left,
    right
};

struct Weighted_labeled_sample{
    explicit Weighted_labeled_sample(const Labeled_sample & ls);
    const std::vector<float> & features;
    const label_t & label;
    float weight;
};

struct DecisionStump{
    unsigned char dimension;
    float threshold;
    direction_t direction;
    float voting_weight;
};

template <typename T>
class Decision_stump_prediction : public DecisionStump{
public:
    std::vector<label_t> classify(const std::vector<T> & data){
        std::vector<label_t> confidence;
        for(auto & datum : data) {
            confidence.emplace_back(classify(datum));
        }
        return confidence;
    }

    /* If direction = left
     * Samples on the left are class0, samples on the right, class1
     * If direction = right
     * Samples on the left are class1, samples on the right, class0
     */
    label_t classify(const T & datum){
        float feature_value{datum.features[dimension]};
        if (feature_value < threshold && direction == direction_t::left) {
            return label_t::class0;
        }
        else if (feature_value >= threshold && direction == direction_t::left) {
            return label_t::class1;
        }
        else if (feature_value < threshold && direction == direction_t::right) {
            return label_t::class1;
        }
        else if (feature_value >= threshold && direction == direction_t::right) {
            return label_t::class0;
        }
    }
};

class Decision_stump_learning : public DecisionStump{
public:
    DecisionStump learn_stump();

private:
    wlabeled_data_t & training_data;
    void sort();
    std::vector<float> compute_cum_sum();
    unsigned char select_dimension();
    direction_t select_direction();
    void update_wclassifier();
    void compute_voting_weight();

    // 1) Initialize weights

    // For k = 1...K
    // 2) Learn a weak classifier

    // Given N training samples and m feature dimensionality
    // 1.- For each of dimension
    // 1.1) Sort samples in ascending order along dimension d
    // 2.2) Compute N weighted cumulative sums
    // 3.1) Select the weighted cumulative sum
    // 2) Select global extremum of all m cumulative sums

    // 3) Compute voting weight
    // 4) Recompute weights

};
