#ifndef NET_H
#define NET_H

#include <vector>
#include <string>
#include <tuple>
#include <armadillo>
#include <iostream>

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
    Net(std::vector<int> dimensions, ClassificationType class_type, ActivationType act_type, double train_rate, double momentum, bool verbose);
    arma::vec forward(const arma::vec inputs);
    void back_prop(const arma::vec expected);
    double get_error(const arma::vec expected);
    void test(DataSet s);
    void train_and_test(DataSet train_data, DataSet test_data, double target, double training_interval);
    std::string to_s();
  private:
    ActivationType act_type;
    ClassificationType class_type;
    std::vector<arma::mat> weights;
    std::vector<arma::mat> del_weights;
    std::vector<arma::vec> layers;
    double train_rate;
    double momentum;
    int input_count;
    std::function<double(double)> activator;
    std::function<arma::vec(arma::vec)> deriverator;
    bool verbose;
    void load_inputs(const arma::vec inputs);
    // std::vector<double> get_outputs(int index);
  };
}

#endif
