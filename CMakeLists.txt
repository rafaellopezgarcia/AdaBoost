cmake_minimum_required(VERSION 3.10)
project(AdaBoost)

set(CMAKE_CXX_STANDARD 17)

add_executable(AdaBoost main.cpp Supervised_learning.cpp Supervised_learning.h AdaBoost.cpp AdaBoost.h Decision_stump.cpp Decision_stump.h)