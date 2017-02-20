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



Net::Net(std::vector<int> dimensions, ClassificationType class_type, ActivationType act_type, double train_rate, double momentum, bool verbose) {
  if(dimensions.size() == 0) {
    throw "Invalid dimensions";
  }
  srand(time(NULL));
  this->train_rate = train_rate;
  this->input_count = dimensions[0];
  this->class_type = class_type;
  this->act_type = act_type;
  this->verbose = verbose;
  this->momentum = momentum;

  switch(act_type) {
    default:
    case Sigmoid:
      this->activator = [](double val) {return 1.0/(1+std::exp(-val));};
      this->deriverator = [](arma::vec val) {arma::vec retval(val); return retval % (1-retval);};
      break;
    case ReLU:
      this->activator = [](double val) {return val > 0 ? val : 0;};
      this->deriverator =[](arma::vec val) {
        arma::vec retval(val);
        retval.transform([](double d) {return d >= 0 ? d : 0;});
        return retval;
      };
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
    }
    layers[i].transform(this->activator);
  }

  return layers.back();
}


void Net::back_prop(const arma::vec expected) {
  if(expected.n_rows != layers.back().n_rows) {
    throw "Incorrect parameter size";
  }

  arma::vec gradients = expected - layers.back();
  gradients %= this->deriverator(layers.back());


  for(int i = layers.size()-2; i >= 0; i--) {
    del_weights[i] = (this->train_rate * (layers[i] * gradients.t())) + (this->momentum * del_weights[i]);
    weights[i] += del_weights[i];
    gradients = weights[i].submat(1,0,weights[i].n_rows-1, weights[i].n_cols-1) * gradients;
    gradients %= this->deriverator(layers[i].rows(1, layers[i].n_rows-1));
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

void Net::test(DataSet s) {
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
  std::cout << std::endl << std::endl << "~~ RESULTS ~~" << std::endl;
  std::cout << correct << "/" << total_count << " correct, " << ((correct + 0.0)/total_count * 100) << "% Accuracy" << std::endl;

}

void Net::train_and_test(DataSet train_data, DataSet test_data, double target, double training_interval) {
  int train_index = 0;
  int test_index = 0;
  int counter = 0;
  while(true) {
    std::cout << "Training " << counter << std::endl;
    double error_sum = 0;
    for(int i = 0; i < training_interval; i++) {
      forward(train_data[train_index].first);
      error_sum += get_error(train_data[train_index].second);
      back_prop(train_data[train_index].second);
      train_index = (train_index + 1) % train_data.size();
      counter++;
    }
    double train_avg = error_sum / training_interval;

    //test with test data
    std::cout << "Testing..." << std::endl;
    forward(test_data[test_index].first);
    double val_avg = get_error(test_data[test_index].second);
    if (val_avg <= target) {
      std::cout << "finished training: " << val_avg << std::endl;
      break;
    }
    else {
      std::cout << "\tTrain Error: " << train_avg << ", Validate Error: " << val_avg << ", Target: " << target << std::endl;
      test_index = (test_index + 1) % test_data.size();
    }
  }
}
