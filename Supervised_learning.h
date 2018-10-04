#pragma once
#include <vector>
#include <iostream>

enum class label_t{
    class0 = -1,
    class1 = 1
};

struct Sample{
    std::vector<float> features;
};

struct Labeled_sample{
    std::vector<float> features;
    label_t label;
};

struct Labeled_sample_printer : public Labeled_sample{
    Labeled_sample_printer(Labeled_sample ls) : Labeled_sample(ls){}

    friend std::ostream &operator<<(std::ostream &os, const Labeled_sample_printer &p) {
        for(auto & f : p.features){
            std::cout << f << " ";
        }
        std::cout << static_cast<unsigned short>(p.label);
        return os;
    }
};

using unlabeled_data_t = std::vector<Sample>;
using labeled_data_t = std::vector<Labeled_sample>;

class Supervised_learning {

};
