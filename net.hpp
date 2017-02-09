#ifndef NET_H
#define NET_H

#include <vector>
#include <string>
#include <tuple>
#include "perceptron.hpp"

namespace net {

  typedef std::vector<std::vector<double>> Weights;
  typedef std::pair<Weights, std::vector<Perceptron*>> Layer;

  enum ActivationType {
    Sigmoid,
    ReLU
  };

  class Net {
  public:
    ~Net();
    Net(std::vector<int> dimensions, ActivationType type);
    std::vector<double> forward(std::vector<double> inputs);
  private:
    std::vector<Layer> layers;
    ActivationType activation_type;
    int input_count;
  };
}

#endif
