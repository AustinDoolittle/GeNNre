#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <armadillo>
#include <vector>
#include <string>
#include <random>
#include "net.hpp"

using namespace net;



Net::Net(std::vector<int> dimensions, ClassificationType class_type, ActivationType act_type, double train_rate, bool verbose) {
  if(dimensions.size() == 0) {
    throw "Invalid dimensions";
  }
  srand(time(NULL));
  this->train_rate = train_rate;
  this->input_count = dimensions[0];
  this->class_type = class_type;
  this->act_type = act_type;
  this->verbose = verbose;

  switch(act_type) {
    default:
    case Sigmoid:
      this->activator = [](double val) {return 1.0/(1+std::exp(-val));};
      this->deriverator = [](double val) {return val * (1-val);};
      break;
    case ReLU:
      this->activator = [](double val) {return -1;};
      this->deriverator =[](double val) {return -1;};
  }

  std::mt19937 engine;  // Mersenne twister random number engine

  std::uniform_real_distribution<double> distr(-0.1, 0.1);

  for(int i = 0; i < dimensions.size(); i++) {
    if(i != dimensions.size()-1) {
      this->weights.push_back(arma::mat(dimensions[i] + 1, dimensions[i+1]));
      this->weights[i].imbue([&](){return distr(engine);});

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
  gradients.transform(this->deriverator);


  for(int i = layers.size()-2; i >= 0; i--) {
    weights[i] += (this->train_rate * (layers[i] * gradients.t()));
    gradients = weights[i].submat(1,0,weights[i].n_rows-1, weights[i].n_cols-1) * gradients;
    gradients.transform(this->deriverator);
  }
}

arma::vec Net::get_error(const arma::vec expected){

  arma::vec outputs = layers.back();

  if(expected.n_rows != outputs.n_rows) {
    throw "Incorrect parameter size";
  }

  arma::vec retVal;

  return .5 * square(expected - outputs);
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

void Net::train(DataSet data ){
  for(int i = 0; i < data.size(); i++) {
    if(this->verbose) {
      std::cout << "Training: " << i << std::endl;
      std::cout << "\tInputs: ";
      for(int j = 0; j < data[i].first.n_rows; j++) {
        std::cout << data[i].first(j) << " ";
      }
      std::cout << std::endl;
    }
    arma::vec result = forward(data[i].first);
    if(this->verbose)
    {
      std::cout << std::endl << "\tOutputs: ";
      for(int j = 0; j < data[i].second.n_rows; j++) {
        std::cout << result[j] << "," << data[i].second(j) << " ";
      }
      std::cout << std::endl;
      arma::vec err = get_error(data[i].second);
      std::cout << "\tErrors: ";
      for(int j = 0; j < err.n_cols; j++) {
        std::cout << err(j) << " ";
      }
      std::cout << std::endl << std::endl;
    }
    back_prop(data[i].second);
  }
}

void Net::test(DataSet s) {
  int total_count = s.size();
  int correct = 0;
  for(int i = 0; i < s.size(); i++) {
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
