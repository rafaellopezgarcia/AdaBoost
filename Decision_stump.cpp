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

Decision_stump_learning::Decision_stump_learning(wlabeled_data_t & training_data):
                         training_data_(training_data),
                         n_training_samples_(training_data_.size())
{
    if (!training_data.empty()){
        n_dim_ = training_data_[0].features.size();
        order_.resize(n_dim_, std::vector<int>(n_training_samples_,0));
        sort();
    }
};

DecisionStump Decision_stump_learning::learn_stump(){
    DecisionStump ds;

    for(unsigned int dim = 0; dim < n_dim_; ++dim) {
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

std::vector<float> Decision_stump_learning::compute_cum_sum(unsigned int dim){
    std::vector<float> cumsums(n_training_samples_, 0);

    /* sort and weight features in dimesion dim */
    std::vector<float> sorted_weighted_feats(n_training_samples_, 0);
    for(unsigned int i = 0; i < n_training_samples_; ++i){
        auto ind = order_[dim][i];
        sorted_weighted_feats[i] = training_data_[ind].features[dim] * training_data_[ind].weight;
    }

    std::partial_sum(sorted_weighted_feats.begin(), sorted_weighted_feats.end(), cumsums.begin());

    /*for (auto it = cumsums.begin(); it != cumsums.end(); ++it){
        std::cout << "- " << *it << std::endl;
    }
    std::cout << training_data_[0].weight << std::endl;*/

}

