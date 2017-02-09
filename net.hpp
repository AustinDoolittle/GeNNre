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
    std::vector<double> forward(const std::vector<double> inputs);
    void back_prop(const std::vector<double> expected);
    std::vector<double> get_error(const std::vector<double> expected);
  private:
    std::vector<Layer> layers;
    std::vector<double> curr_inputs;
    ActivationType activation_type;
    int input_count;
  };
}

#endif
