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



Net::Net(std::vector<int> dimensions, ClassificationType class_type, ActivationType act_type, double momentum, bool verbose) {
  if(dimensions.size() == 0) {
    throw "Invalid dimensions";
  }
  srand(time(NULL));
  this->train_rate = DEF_TRAIN_RATE;
  this->input_count = dimensions[0];
  this->class_type = class_type;
  this->act_type = act_type;
  this->verbose = verbose;
  this->momentum = momentum;

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

void Net::load_inputs(const arma::vec vector) {
  if(vector.n_rows != input_count) {
    std::cerr << "Parameter size mismatch" << std::endl;
    throw;
  }
  this->layers[0].rows(1,layers[0].n_rows - 1) = vector;
}

arma::vec Net::forward(const arma::vec inputs) {

  load_inputs(inputs);

  for (int i = 1; i < this->layers.size(); i++) {
    if(i != this->layers.size() - 1) {
      layers[i].rows(1, layers[i].n_rows - 1) = weights[i-1].t() * layers[i-1];

    }
    else {
      layers[i] = weights[i-1].t() * layers[i-1];
      if(this->act_type == ReLU) {
        layers[i].transform(SIG_ACT);
        break;
      }
    }

    layers[i].transform(this->activator);
  }

  return layers.back();
}


void Net::back_prop(const arma::vec expected) {
  if(expected.n_rows != layers.back().n_rows) {
    throw "Incorrect parameter size";
  }

  arma::vec back(layers.back());
  arma::vec gradients = expected - back;
  if(this->act_type == ReLU) {
    gradients %= back % (1-back);
  }
  else {
    gradients %= deriverator(back);
  }

  prev_del_weights = del_weights;
  for(int i = layers.size()-2; i >= 0; i--) {
    del_weights[i] = (this->train_rate * (layers[i] * gradients.t())) + (this->momentum * del_weights[i]);
    weights[i] += del_weights[i];
    gradients = weights[i].submat(1,0,weights[i].n_rows-1, weights[i].n_cols-1) * gradients;
    gradients %= this->deriverator(layers[i].rows(1, layers[i].n_rows-1));
  }
}

arma::vec Net::deriverator(arma::vec v) {
  switch(this->act_type){
    case Sigmoid:
      return v % (1-v);
    case ReLU:
      return v.transform([](double d) {return d > 0 ? 1 : LEAKY_RELU_CONST;});
    case TanH:
      return v.transform([](double d) {return 1 - std::pow((2.0 / (1.0 + std::exp(-2.0 * d))) - 1, 2);});
    default:
      return v;
  }
}

double Net::get_error(const arma::vec expected){

  arma::vec outputs = layers.back();

  if(expected.n_rows != outputs.n_rows) {
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
    arma::vec result = forward(s[i].first);
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

void Net::train_and_test(DataSet train_data, DataSet test_data, double target, double training_interval) {
  int train_index = 0;
  int test_index = 0;
  double prev_err = .5;

  int counter = 0;
  while(true) {
    if(this->verbose) {
      std::cout << "Training " << counter << std::endl;
    }
    double error_sum = 0;
    for(int i = 0; i < training_interval; i++) {
      arma::vec res = forward(train_data[train_index].first);
      if(this->verbose) {
        std::cout << "Outputs/Expected: ";
        for(int j = 0; j < res.n_rows; j++) {
          std::cout << res(j) << "/" << train_data[train_index].second(j) << " ";
        }
        std::cout << std::endl;
      }
      double err = get_error(train_data[train_index].second);
      error_sum += err;


      back_prop(train_data[train_index].second);
      train_index = (train_index + 1) % train_data.size();

      //Change train rate at the end of an epoch
      if(train_index == 0) {
        if(err < prev_err) {
          this->train_rate *= 1.05;
        }
        else if(err > (prev_err + ERR_CHANGE)) {
          rollback_weights();
          this->train_rate *= .5;
        }
        prev_err = err;
      }
      counter++;
    }
    double train_avg = error_sum / training_interval;

    //test with test data
    if(this->verbose) {
      std::cout << "Testing..." << std::endl;
    }
    forward(test_data[test_index].first);
    double val_avg = get_error(test_data[test_index].second);
    if (val_avg <= target) {
      if(this->verbose) {
        std::cout << "finished training: " << val_avg << std::endl;
      }
      break;
    }
    else {
      if(this->verbose) {
        std::cout << "\tTrain Error: " << train_avg << ", Validate Error: " << val_avg << ", Target: " << target << std::endl;
      }
      test_index = (test_index + 1) % test_data.size();
    }
    if(counter > 2000000) {
      break;
    }
  }
}
