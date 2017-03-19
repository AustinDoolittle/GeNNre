#ifndef NET_H
#define NET_H

#include <vector>
#include <string>
#include <math.h>
#include <tuple>
#include <armadillo>
#include <iostream>

#define LEAKY_RELU_CONST .01
#define RELU_THRESH 1
#define ERR_CHANGE .00000000001
#define DEF_TRAIN_RATE 0.5
#define DEF_DROPOUT 0.5
#define DEF_DROPOUT_INP 0.9
#define TIMEOUT_LENGTH 18000
#define DEF_VAL_INTERVALS 5


#define SIG_ACT [](double val) {return 1.0/(1.0 + std::exp(-val));}
#define RELU_ACT [](double val) {return val > (LEAKY_RELU_CONST * val) ? val : (LEAKY_RELU_CONST * val);}
#define TANH_ACT [](double val) {return (2.0 / (1.0 + std::exp(-2.0 * val))) - 1;}
#define RELU_CLIPPER [](double val) {return (std::abs(val) > RELU_THRESH) ? val * (RELU_THRESH / std::abs(val)) : val;}
#define DROPOUT_ALG [](double val) {return val * random_from_prob(DEF_DROPOUT);}
#define DROPOUT_INP_ALG [](double val) {return val * random_from_prob(DEF_DROPOUT_INP);}
#define DROPOUT_FORWARD_ALG [](double val) {return val * DEF_DROPOUT;}
#define DROPOUT_INP_FORWARD_ALG [](double val) {return val * DEF_DROPOUT_INP;}

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
    Net(std::vector<int> dimensions, ClassificationType class_type, ActivationType act_type, double momentum, bool verbose, bool is_dropout);
    arma::vec forward_train(const arma::vec inputs);
    arma::vec forward_test(const arma::vec inputs);
    void back_prop(const arma::vec expected);
    double get_error(const arma::vec expected);
    double test(DataSet s);
    void train_and_test(DataSet train_data, DataSet test_data, double target, double training_interval, int diverge_count);
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
    static int random_from_prob(double prob);
    double train_rate;
    double momentum;
    int input_count;
    std::function<double(double)> activator;
    std::function<double(double)> relu_clipper;
    bool verbose;
    bool dropout;

    void load_inputs(const arma::vec inputs);
    // std::vector<double> get_outputs(int index);
  };
}

#endif
