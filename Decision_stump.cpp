#include <algorithm> /*implements sort */
#include <numeric> /*implements partial_sum */
#include "Decision_stump.h"


Weighted_labeled_sample::Weighted_labeled_sample(Labeled_sample & ls) : features(ls.features),
                                                                              label(ls.label)
{
}

Sorting_sample::Sorting_sample(const std::vector<float> &features, unsigned int ind) : features(features),
                                                                                       ind(ind)
{
}

Sorting_sample &Sorting_sample::operator=(const Sorting_sample & rhs){
    if (this != &rhs){
        ind = rhs.ind;
    }
    return *this;
}

DecisionStump::DecisionStump() {}

DecisionStump::DecisionStump(unsigned char dimension,
                             float threshold,
                             direction_t direction) : dimension(dimension),
                                                      threshold(threshold),
                                                      direction(direction) {

}

Decision_stump_learning::Decision_stump_learning(wlabeled_data_t & training_data):
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

    for(unsigned char dim = 0; dim < n_dim_; ++dim) {
        compute_cum_sum(dim);
    }

}

void Decision_stump_learning::sort(){
    /* declare and fill sorting_vector */
    std::vector<Sorting_sample> sorting_vector;
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

/* Total sum
    1 1 0 0 1 1
    w1, w2, w3, w4, w5, w6

    a= w1+w2+w5+w6
    b= w3+w4

    a = w2+w5+w6
    b = w1+w3+w4 = all - a



    */
std::vector<float> Decision_stump_learning::compute_cum_sum(unsigned char dim){
    std::vector<float> sum_left(n_training_samples_, 0);
    std::vector<float> sum_right(n_training_samples_, 0);

    DecisionStump ds;
    ds.dimension = dim;
    ds.direction = direction_t::left;
    for (unsigned int i = 0; i < n_training_samples_; ++i){
        const auto & sample = training_data_[order_[dim][i]];
        ds.threshold = sample.features[dim];
        if (I(sample, ds) == 1){
            sum_left[0] += sample.weight;
        }
        else{
            sum_right[0] += sample.weight;
        }
    }

    for (unsigned int i = 0; i < n_training_samples_-1; ++i){
        const auto & sample = training_data_[order_[dim][i]];
        ds.threshold = training_data_[order_[dim][i+1]].features[dim];
        if (I(sample, ds) == 1){
            sum_left[i+1]=sum_left[i]+sample.weight;
            sum_right[i+1]=sum_right[i]-sample.weight;
        }
        else{
            sum_left[i+1]=sum_left[i]-sample.weight;
            sum_right[i+1]=sum_right[i]+sample.weight;
        }

    }

    std::cout<<"dimension "<<dim<<std::endl;
    for(unsigned int i=0; i<sum_left.size(); ++i){
        const auto & sample = training_data_[order_[dim][i]];
        std::cout<<sample.features[dim]<<" ";
    }
    for(unsigned int i=0; i<sum_left.size(); ++i){
        const auto & sample = training_data_[order_[dim][i]];
        std::cout<<static_cast<int>(sample.label)<<" ";
    }
    std::cout<<std::endl;
    for(unsigned int i=0; i<sum_left.size(); ++i){
        std::cout<<sum_left[i]<<" ";
    }
    std::cout<<std::endl;
    for(unsigned int i=0; i<sum_right.size(); ++i){
        std::cout<<sum_right[i]<<" ";
    }
    std::cout<<std::endl;

    return sum_left;

}

unsigned short Decision_stump_learning::I(const Weighted_labeled_sample & sample,
                                          const DecisionStump ds){

    Decision_stump_prediction<const Weighted_labeled_sample> predictor(ds);
    if (predictor.classify(sample) == sample.label){
        return 1u;
    }

    else{
        return 0u;
    }
}


