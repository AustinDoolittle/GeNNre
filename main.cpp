#include <vector>
#include <iostream>
#include <tuple>
#include "net.hpp"

#define NUM_OUTPUTS 1
#define NUM_INPUTS 2

using namespace net;


int main() {
  std::vector<int> layout = {NUM_INPUTS, 2,  NUM_OUTPUTS};
  TrainingData training_sets;
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 2; j++) {
      training_sets.push_back(std::make_pair(std::vector<double>{(double)i, (double)j}, std::vector<double>{(double)(i | j)}));
    }
  }

  Net net(layout, Sigmoid);
  net.to_s();
  for(int i = 0; i < 1000; i++) {
    net.train(training_sets);
  }
  net.to_s();
}
