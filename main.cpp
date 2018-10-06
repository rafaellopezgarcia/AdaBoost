#include <iostream>
#include "AdaBoost.h"
#include <numeric>
/* AdaBoost algorithm
 * Learning
 * It learns a set of weak classifiers
 * Inference
 * f(x_i) = sum( alpha_k*h_k(x_i) )
 * H(x_i) = sign(f(x_i))
 */


 /*
  * Training data: vector of (Features, Labels)
  * Features (theta1, theta2, ..., theta3)
  * Labels (class0, class1)
  *
  */


// sign, dimension, voting weight

struct A{
    std::vector <int> v {1,65,12,34};

};
int main() {
    labeled_data_t data;
    Labeled_sample ls{std::vector<float>{1.f,10.f,30.f}, label_t::class0};
    Labeled_sample ls2{std::vector<float>{4.f,7.f,200.f}, label_t::class1};
    Labeled_sample ls3{std::vector<float>{400.f,400.f,400.f}, label_t::class1};
    data.emplace_back(ls);
    data.emplace_back(ls2);
    data.emplace_back(ls3);

    adaboost::Learning learning(1);
    learning.train_model(data);


    std::vector <int> v {1,4,12,20};
    std::vector <int> result(v.size(), 0);
    int acc=0;
    auto vv{20u};
    std::partial_sum(v.begin(), v.end(), result.begin(), [&vv](auto &a, auto &b){ return a + vv*b; });
    for(auto it = result.begin(); it != result.end(); ++it)
        std::cout << *it << std::endl;
    //std::cout << v.size() << " " << acc << std::endl;



    return 0;
}