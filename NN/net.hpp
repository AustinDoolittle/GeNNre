#ifndef NET_H
#define NET_H

#include <vector>
#include <string>
#include <math.h>
#include <tuple>
#include <armadillo>
#include <iostream>

#define LEAKY_RELU_CONST .01
#define ERR_CHANGE .00000000001
#define DEF_TRAIN_RATE 0.5


#define SIG_ACT [](double val) {return 1.0/(1.0 + std::exp(-val));}
#define RELU_ACT [](double val) {return val > (LEAKY_RELU_CONST * val) ? val : (LEAKY_RELU_CONST * val);}
#define TANH_ACT [](double val) {return (2.0 / (1.0 + std::exp(-2.0 * val))) - 1;}

namespace net {


  typedef std::vector<std::pair<arma::vec,arma::vec>> DataSet;

  enum ActivationType {
    Sigmoid,
    ReLU,
    TanH
  };

  enum ClassificationType {
    Single,
    Multi
  };

  class Net {
  public:
    ~Net();
    Net(std::vector<int> dimensions, ClassificationType class_type, ActivationType act_type, double momentum, bool verbose);
    arma::vec forward(const arma::vec inputs);
    void back_prop(const arma::vec expected);
    double get_error(const arma::vec expected);
    double test(DataSet s);
    void train_and_test(DataSet train_data, DataSet test_data, double target, double training_interval);
    std::string to_s();
  private:
    ActivationType act_type;
    ClassificationType class_type;
    std::vector<arma::mat> weights;
    std::vector<arma::mat> del_weights;
    std::vector<arma::mat> prev_del_weights;
    std::vector<arma::vec> layers;
    void rollback_weights();
    arma::vec deriverator(arma::vec v);
    double train_rate;
    double momentum;
    int input_count;
    std::function<double(double)> activator;
    bool verbose;
    void load_inputs(const arma::vec inputs);
    // std::vector<double> get_outputs(int index);
  };
}

#endif
