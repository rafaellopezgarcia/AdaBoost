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
    A(int a, int b) : a(a), b(b) {}

    int a,b;


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
    std::vector<A> a;
    a.emplace_back(2,4);
    a.emplace_back(2,4);
    a.emplace_back(3,7);
    a.emplace_back(4,42);
    a.emplace_back(4,9);
    a.emplace_back(4,3);
    a.emplace_back(5,1);
    a.emplace_back(5,3);
    a.emplace_back(6,14);
    a.emplace_back(6,24);

    auto right_value=a.begin()->b;
    auto it_left=a.begin();
    for(auto it_right=a.begin(); it_right!=a.end()-1; ++it_right){
        if(it_right->a!=(it_right+1)->a){
            for(auto it2=it_left; it2!=it_right+1; ++it2){
                it2->b=right_value;
            }
            right_value=(it_right+1)->b;
            it_left=it_right+1;
        }
        else if(it_right==a.end()-2){
            for(auto it2=it_left; it2!=a.end(); ++it2){
                it2->b=right_value;
            }
        }
    }


    for(auto it=a.begin(); it!=a.end(); ++it){
        std::cout<<it->a<<" "<<it->b<<std::endl;
    }


    return 0;
}