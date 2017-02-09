#include <vector>
#include <iostream>
#include "net.hpp"

#define NUM_OUTPUTS 1
#define NUM_INPUTS 2

using namespace net;

int main() {
  std::vector<int> layout = {NUM_INPUTS, 3, NUM_OUTPUTS};
  Net net(layout, Sigmoid);
  std::vector<double> retVal = net.forward(std::vector<double>{2, 3});
  for(double d : retVal) {
    std::cout << "value: " << d << std::endl;
  }

  retVal = net.get_error(std::vector<double>{0});
  for(double d : retVal) {
    std::cout << "error: " << d << std::endl;
  }
}
