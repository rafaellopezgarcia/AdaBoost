#include "Decision_stump.h"

Weighted_labeled_sample::Weighted_labeled_sample(const Labeled_sample & ls) : features(ls.features),
                                                                              label(ls.label)
{

}