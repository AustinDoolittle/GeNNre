#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <armadillo>
#include <math.h>
#include <vector>
#include <string>
#include <random>
#include "net.hpp"

using namespace net;



Net::Net(std::vector<int> dimensions, ClassificationType class_type, ActivationType act_type, double momentum, bool verbose, bool is_dropout) {
  if(dimensions.size() == 0) {
    std::cerr << "Invalid Dimensions" << std::endl;
    throw "Invalid dimensions";
  }
  srand(time(NULL));
  this->train_rate = DEF_TRAIN_RATE;
  this->input_count = dimensions[0];
  this->class_type = class_type;
  this->act_type = act_type;
  this->verbose = verbose;
  this->momentum = momentum;
  this->dropout = is_dropout;
  this->relu_clipper = RELU_CLIPPER;

  switch(act_type) {
    default:
    case Sigmoid:
      this->activator = SIG_ACT;
      break;
    case ReLU:
      this->activator = RELU_ACT;
      break;
    case TanH:
      this->activator = TANH_ACT;
      break;
  }

  std::mt19937 engine;  // Mersenne twister random number engine

  std::uniform_real_distribution<double> distr(-0.1, 0.1);

  for(int i = 0; i < dimensions.size(); i++) {
    if(i != dimensions.size()-1) {
      this->weights.push_back(arma::mat(dimensions[i] + 1, dimensions[i+1]));
      this->weights[i].imbue([&](){return distr(engine);});
      this->del_weights.push_back(arma::mat(dimensions[i] + 1, dimensions[i+1], arma::fill::zeros));
      this->layers.push_back(arma::vec(dimensions[i] + 1, arma::fill::ones));
    }
    else {
      this->layers.push_back(arma::vec(dimensions[i], arma::fill::ones));
    }
  }
}

Net::~Net() {

}

int Net::random_from_prob(double prob) {
  double rndDouble = (double)rand() / RAND_MAX;
  return rndDouble < prob;
}

void Net::load_inputs(const arma::vec vector) {
  if(vector.n_rows != input_count) {
    std::cerr << "Parameter size mismatch" << std::endl;
    throw;
  }

  this->layers[0].rows(1,layers[0].n_rows - 1) = vector;
}

arma::vec Net::forward_train(const arma::vec inputs) {

  load_inputs(inputs);

  if (this->dropout) {
    this->layers[0].rows(1,layers[0].n_rows - 1).transform(DROPOUT_INP_ALG);
  }

  for (int i = 1; i < this->layers.size(); i++) {
    if(i != this->layers.size() - 1) {
      layers[i].rows(1, layers[i].n_rows - 1) = weights[i-1].t() * layers[i-1];
      if(this->dropout)  {
        layers[i].rows(1, layers[i].n_rows - 1).transform(DROPOUT_ALG);
      }
    }
    else {
      layers[i] = weights[i-1].t() * layers[i-1];
      if(this->dropout)  {
        layers[i].transform(DROPOUT_ALG);
      }
    }

    layers[i].transform(this->activator);

  }

  return layers.back();
}

arma::vec Net::forward_test(const arma::vec inputs) {

  load_inputs(inputs);

  if (this->dropout) {
    this->layers[0].rows(1,layers[0].n_rows - 1).transform(DROPOUT_INP_FORWARD_ALG);
  }

  for (int i = 1; i < this->layers.size(); i++) {
    if(i != this->layers.size() - 1) {
      layers[i].rows(1, layers[i].n_rows - 1) = weights[i-1].t() * layers[i-1];
      if(this->dropout)  {
        layers[i].rows(1, layers[i].n_rows - 1).transform(DROPOUT_FORWARD_ALG);
      }
    }
    else {
      layers[i] = weights[i-1].t() * layers[i-1];
      if(this->dropout)  {
        layers[i].transform(DROPOUT_FORWARD_ALG);
      }
    }

    layers[i].transform(this->activator);

  }

  return layers.back();
}


void Net::back_prop(const arma::vec expected) {
  if(expected.n_rows != layers.back().n_rows) {
    std::cerr << "Incorrect Parameter Size (back_prop)" << std::endl;
    throw "Incorrect parameter size";
  }

  arma::vec back(layers.back());
  arma::vec gradients = expected - back;
  gradients %= deriverator(back);

  prev_del_weights = del_weights;
  for(int i = layers.size()-2; i >= 0; i--) {
    //check if this is relu
    if(this->act_type == ReLU) {
      //it is, clip the gradients
      gradients.transform(this->relu_clipper);
    }
    del_weights[i] = (this->train_rate * (layers[i] * gradients.t())) + (this->momentum * del_weights[i]);
    weights[i] += del_weights[i];
    gradients = weights[i].submat(1,0,weights[i].n_rows-1, weights[i].n_cols-1) * gradients;
    arma::vec copy(layers[i].rows(1, layers[i].n_rows-1));
    gradients %= this->deriverator(copy);
  }
}

arma::vec Net::deriverator(arma::vec v) {
  switch(this->act_type){
    case Sigmoid:
      return v % (1-v);
    case ReLU:
      return v.transform([](double d) {return d > 0 ? 1 : LEAKY_RELU_CONST;});
    case TanH:
      return 1 - arma::pow((2.0 / (1.0 + arma::exp(-2.0 * v))) - 1, 2);
    default:
      return v;
  }
}

double Net::get_error(const arma::vec expected){

  arma::vec outputs = layers.back();

  if(expected.n_rows != outputs.n_rows) {
    std::cerr << "Incorrect parameter size (get_error)" << std::endl;
    throw "Incorrect parameter size";
  }

  return arma::sum(arma::square(expected - outputs)) / expected.n_rows;
}

std::string Net::to_s() {
  std::string retval = "";
  for(int i = 0; i < layers.size(); i++) {
    retval += "Layer " + std::to_string(i) + ":\n";
    for(int j = 0; j < layers[i].n_rows; j++) {
      retval += "\tNode " + std::to_string(j) + ": " + std::to_string(layers[i](j)) + "\n";
      if (i != layers.size() - 1){
        arma::rowvec row_weight = weights[i].row(j);
        for(int k = 0; k < row_weight.n_cols; k++) {
          retval += "\t\tWeight " + std::to_string(k) + ": " + std::to_string(row_weight(k)) + "\n";
        }
      }
    }
  }

  return retval;
}

double Net::test(DataSet s) {
  int total_count = s.size();
  int correct = 0;
  for(int i = 0; i < total_count; i++) {
    arma::vec result = forward_test(s[i].first);
    if(this->verbose) {
      std::cout << "Testing: " << i << std::endl;
      std::cout << "\tOutput/Expected: ";
      for(int r = 0; r < result.n_rows; r++) {
        std::cout << result(r) << "/" << s[i].second(r) << " ";
      }
      std::cout << std::endl;
    }
    if (this->class_type == Single) {
      int max_index = 0;
      int expected_index = 0;
      for(int j = 1; j < result.size(); j++) {
        if (result(j) > result(max_index)) {
          max_index = j;
        }
        if(s[i].second(j) > s[i].second(expected_index)) {
          expected_index = j;
        }
      }
      if(max_index == expected_index) {
        correct++;
        if(this->verbose) {
          std::cout << "\tCorrect\t" << correct << "/" << (i + 1) << std::endl;
        }
      }
      else {
        if(this->verbose) {
          std::cout << "\tWrong\t" << correct << "/" << (i + 1) << std::endl;
        }
      }
    }
    else if(this->class_type == Multi) {
      //TODO: implement
    }
  }
  if(this->verbose) {
    std::cout << std::endl << std::endl << "~~ RESULTS ~~" << std::endl;
    std::cout << correct << "/" << total_count << " correct, " << ((correct + 0.0)/total_count * 100) << "% Accuracy" << std::endl;
  }
  return (correct + 0.0)/total_count;
}

void Net::rollback_weights() {
  for(int i = 0; i < weights.size(); i++) {
    weights[i] -= del_weights[i];
  }
  del_weights = prev_del_weights;
}

void Net::train_and_test(DataSet train_data, DataSet test_data, double target, double training_interval, int diverge_limit) {
  int test_index = 0;
  double prev_val = .5;
  int diverge_count = 0;
  std::clock_t ts, te;
  ts = clock();

  int counter = 0;
  while(true) {
    if(this->verbose) {
      std::cout << "Training " << counter << std::endl;
    }
    double error_sum = 0;
    double prev_err = .5;

    for(int i = 0; i < training_interval; i++) {
      error_sum = 0;
      for (int j = 0; j < train_data.size(); j++) {
        arma::vec res = forward_train(train_data[j].first);
        if(this->verbose) {
          std::cout << "Outputs/Expected: ";
          for(int k = 0; k < res.n_rows; k++) {
            std::cout << res(k) << "/" << train_data[j].second(k) << " ";
          }
          std::cout << std::endl;
        }
        error_sum += get_error(train_data[j].second);

        back_prop(train_data[j].second);
      }

      double err_avg = error_sum / train_data.size();

      if(err_avg < prev_err) {
        this->train_rate *= 1.05;
      }
      else if(err_avg > (prev_err + ERR_CHANGE)) {
        rollback_weights();
        this->train_rate *= .5;
      }

      prev_err = err_avg;
      counter++;
    }

    //test with test data
    if(this->verbose) {
      std::cout << "Testing..." << std::endl;
    }
    forward_test(test_data[test_index].first);
    double val_avg = get_error(test_data[test_index].second);
    if (val_avg <= target) {
      if(this->verbose) {
        std::cout << "Finished Training: " << val_avg << std::endl;
      }
      break;
    }
    else {
      if(this->verbose) {
        std::cout << "\tValidate Error: " << val_avg << ", Target: " << target << std::endl;
      }
      test_index = (test_index + 1) % test_data.size();
    }
    if (val_avg > prev_val) {
      diverge_count += 1;
      if (diverge_count >= diverge_limit) {
        break;
      }
    }
    else {
      diverge_count = 0;
    }
    prev_val = val_avg;

    te = clock();

    if (((float)te - (float)ts) > (CLOCKS_PER_SEC * TIMEOUT_LENGTH)) {
      if(this->verbose) {
        std::cout << "~~ TIMEOUT ~~" << std::endl;
      }
    }
  }
}
