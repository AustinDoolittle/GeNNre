#include <vector>
#include <iostream>
#include "net.hpp"

#define NUM_OUTPUTS 1
#define NUM_INPUTS 2

using namespace net;

int main() {
  std::vector<int> layout = {NUM_INPUTS,  NUM_OUTPUTS};
  Net net(layout, Sigmoid);
  std::vector<double> retVal = net.forward(std::vector<double>{1, 2});
  for(double d : retVal) {
    std::cout << "value: " << d << std::endl;
  }
}
