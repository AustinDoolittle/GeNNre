#ifndef NET_H
#define NET_H

#include <vector>
#include <string>
#include <math.h>
#include <tuple>
#include <armadillo>
#include <iostream>
#include "perceptron.hpp"

namespace net {


  typedef std::vector<std::pair<arma::vec,arma::vec>> DataSet;

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
    Net(std::vector<int> dimensions, ClassificationType class_type, ActivationType act_type, double train_rate, bool verbose);
    arma::vec forward(const arma::vec inputs);
    void back_prop(const arma::vec expected);
    arma::vec get_error(const arma::vec expected);
    void train(DataSet data);
    void test(DataSet data);
    std::string to_s();
  private:
    ActivationType act_type;
    ClassificationType class_type;
    std::vector<arma::mat> weights;
    std::vector<arma::vec> layers;
    double train_rate;
    int input_count;
    std::function<double(double)> activator;
    std::function<double(double)> deriverator;
    bool verbose;
    void load_inputs(const arma::vec inputs);
    // std::vector<double> get_outputs(int index);
  };
}

#endif
