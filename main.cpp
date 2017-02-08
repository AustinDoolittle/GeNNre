#include <vector>
#include <iostream>
#include "net.hpp"

#define NUM_OUTPUTS 1
#define NUM_INPUTS 2

using namespace net;

int main() {
  std::vector<int> layout = {NUM_INPUTS, 3, NUM_OUTPUTS};
  Net net(layout, Sigmoid);

}
