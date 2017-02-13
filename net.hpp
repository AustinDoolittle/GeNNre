#ifndef NET_H
#define NET_H

#include <vector>
#include <string>
#include <tuple>
#include "perceptron.hpp"

namespace net {

  typedef std::vector<std::vector<double>> Weights;
  typedef std::vector<Perceptron*> Layer;
  typedef std::vector<std::pair<std::vector<double>,std::vector<double>>> DataSet;

  enum ActivationType {
    Sigmoid,
    ReLU
  };

  enum ClassificationType {
    Single,
    Multi
  };

  class Net {
  public:
    ~Net();
    Net(std::vector<int> dimensions, ClassificationType class_type, ActivationType act_type, double train_rate);
    std::vector<double> forward(const std::vector<double> inputs);
    void back_prop(const std::vector<double> expected);
    std::vector<double> get_error(const std::vector<double> expected);
    void train(DataSet data);
    void test(DataSet data);
    void to_s();
  private:
    ActivationType act_type;
    ClassificationType class_type;
    std::vector<Layer> layers;
    std::vector<Weights> weights_arr;
    std::vector<double> inputs;
    double train_rate;
    int input_count;
    void load_inputs(const std::vector<double> inputs);
    std::vector<double> get_outputs(int index);
  };
}

#endif
