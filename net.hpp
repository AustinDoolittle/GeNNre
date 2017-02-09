#ifndef NET_H
#define NET_H

#include <vector>
#include <string>
#include <tuple>
#include "perceptron.hpp"

namespace net {

  typedef std::vector<std::vector<double>> Weights;
  typedef std::vector<Perceptron*> Layer;
  typedef std::vector<std::pair<std::vector<double>,std::vector<double>>> TrainingData;

  enum ActivationType {
    Sigmoid,
    ReLU
  };

  class Net {
  public:
    ~Net();
    Net(std::vector<int> dimensions, ActivationType type = Sigmoid, double train_rate = 0.5);
    std::vector<double> forward(const std::vector<double> inputs);
    void back_prop(const std::vector<double> expected);
    std::vector<double> get_error(const std::vector<double> expected);
    void train(TrainingData data);
    void to_s();
  private:
    std::vector<Layer> layers;
    std::vector<Weights> weights_arr;
    ActivationType activation_type;
    double train_rate;
    void load_inputs(const std::vector<double> inputs);
    std::vector<double> get_outputs(int index);
  };
}

#endif
