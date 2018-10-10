#pragma once
#include <vector>
#include <ostream>
#include "Supervised_learning.h"

/*nomenclature
 * u->unidimensional
 * w->weighted
 * l->labeled
 * s->sample
 * t->training
 * ds->dataset
 * */

struct WLSample_t;
struct UWLSample_t;
struct DecisionStump;

using WLData_t=std::vector<WLSample_t>;
using UWLData_t=std::vector<UWLSample_t>;

enum class direction_t{
    left,
    right
};

/*Weighted Labeled Sample*/
struct WLSample_t{
    explicit WLSample_t(Labeled_sample & ls);
    ~WLSample_t()=default;
    WLSample_t &operator=(const WLSample_t & rhs) = delete;

    const std::vector<float> &features;
    const label_t & label;
    float weight;
};

/*Unidimensional Weighted Labeled Sample*/
struct UWLSample_t{
    UWLSample_t(const WLSample_t & wls, unsigned short dim);
    UWLSample_t();
    float feature;
    label_t label;
    float weight;
    unsigned short dim;
    float cumsum;

    friend std::ostream &operator<<(std::ostream &os, const UWLSample_t &sample);
};

/*Sorting sample*/
struct SortingSample_t{
    explicit SortingSample_t(const std::vector<float> &features, unsigned int ind);
    SortingSample_t &operator=(const SortingSample_t & rhs);
    const std::vector<float> &features;
    unsigned int ind;
};

struct DecisionStump{
    DecisionStump(unsigned char dimension, float threshold, direction_t direction);
    DecisionStump()= default;
    unsigned short dimension;
    float threshold;
    direction_t direction;
    float voting_weight;
};

/* If direction = left
 * Samples on the left are class0, samples on the right, class1
 * If direction = right
 * Samples on the left are class1, samples on the right, class0
 */
template <typename T>
class Decision_stump_prediction : private DecisionStump{
public:

    explicit Decision_stump_prediction(const DecisionStump & ds) : DecisionStump(ds) {}

    std::vector<label_t> classify(const std::vector<T> & data){
        std::vector<label_t> confidence;
        for(auto & datum : data) {
            confidence.emplace_back(classify(datum));
        }
        return confidence;
    }

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
        else {
            return label_t::class0;
        }
    }
};

class Decision_stump_learning{
public:
    explicit Decision_stump_learning(WLData_t & training_data);
    ~Decision_stump_learning()= default;
    /*learn a weak classifier*/
    DecisionStump learn_stump();

private:
    WLData_t & training_data_;
    std::vector<std::vector<unsigned int>> order_;
    unsigned short n_dim_;
    unsigned long n_training_samples_;

    UWLData_t create_unidimensional_set(unsigned short dim);
    /*sort samples in ascending order along all dimensions*/
    void sort();
    /*compute N weighted cumulative sums*/
    void compute_cum_sum(UWLData_t &uwl_data);
    void update_optimal_stump(DecisionStump &ds,UWLData_t &uwl_data, float &max_cumsum);
};
