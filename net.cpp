#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include "net.hpp"

using namespace net;

Net::Net(std::vector<int> dimensions, ActivationType type, double train_rate) {
  if(dimensions.size() == 0) {
    throw "Invalid dimensions";
  }
  srand(time(NULL));
  this->activation_type = type;
  this->train_rate = train_rate;
  this->input_count = dimensions[0];

  for(int i = 1; i < dimensions.size(); i++) {

    //Weights w(dimensions[i-1] + 1, std::vector<double>(dimensions[i], (rand() % 3 - 1)));
    Weights w(dimensions[i-1] + 1, std::vector<double>(dimensions[i], 1));

    this->weights_arr.push_back(w);

    Layer temp;
    for(int j = 0; j < dimensions[i]; j++) {
      if(type == Sigmoid) {
        temp.push_back(new SigmoidPerceptron());
      }
      else if(type == ReLU) {
        temp.push_back(new ReLUPerceptron());
      }
    }

    this->layers.push_back(temp);
  }
}

Net::~Net() {
  for(int i = 0; i < layers.size(); i++) {
    for(int j = 0; j < layers[i].size(); j++) {
      delete layers[i][j];
    }
  }
}

std::vector<double> Net::get_outputs(int index) {
  if(index >= layers.size() || index < 0) {
    std::cout << "Index out of bounds" << std::endl;
    throw;
  }

  std::vector<double> retVal;
  for(int i = 0; i < layers[index].size(); i++) {
    retVal.push_back(layers[index][i]->get_output());
  }

  //Add a bias if this is not the last layer
  if(index != (layers.size() - 1)) {
    retVal.push_back(1);
  }
  return retVal;
}

void Net::load_inputs(const std::vector<double> vector) {
  if(vector.size() != input_count) {
    std::cerr << "Parameter size mismatch" << std::endl;
    throw;
  }
  this->inputs = vector;
}

std::vector<double> Net::forward(const std::vector<double> inputs) {
  if(inputs.size() != input_count) {
    std::cerr << "Incorrect inputs dimensions" << std::endl;
    throw;
  }

  load_inputs(inputs);

  std::vector<double> v1 = inputs;
  v1.push_back(1); //for bias

  for(int i = 0; i < layers.size(); i++) {
    Weights* w = &weights_arr[i];
    for(int j = 0; j < layers[i].size(); j++) {
      std::vector<double> edges;
      for(int k = 0; k < (*w).size(); k++) {
        edges.push_back((*w)[k][j]);
      }

      layers[i][j]->forward(v1, edges);
    }
    v1 = get_outputs(i);
  }

  return v1;
}


void Net::back_prop(const std::vector<double> expected) {

  if(expected.size() != layers.back().size()) {
    throw "Incorrect parameter size";
  }

  for(int i = layers.size() - 1; i >= 0; i--) {
    for(int j = 0; j < layers[i].size(); j++) {
      double val = layers[i][j]->get_output();
      double grad = 0;
      if(i == layers.size() - 1) {
        grad = (expected[j] - val);
      }
      else {
        for(int k = 0; k < weights_arr[i+1][j].size(); k++) {
          grad += weights_arr[i+1][j][k] * layers[i+1][k]->get_grad();
        }
      }
      grad *= val * (1 - val);
      layers[i][j]->set_grad(grad);
    }

    for(int w = 0; w < weights_arr[i].size(); w++) {
      for(int e = 0; e < weights_arr[i][w].size(); e++) {

        double val;
        if(w == weights_arr[i].size() - 1) {
          val = 1;
        }
        else {
          if(i == 0) {
            //Pull val from inputs
            val = inputs[w];
          }
          else {
            val = layers[i-1][w]->get_output();
          }
        }

        weights_arr[i][w][e] += this->train_rate * layers[i][e]->get_grad() * val;
      }
    }
  }
}

std::vector<double> Net::get_error(const std::vector<double> expected){

  Layer* outputs = &layers[layers.size() - 1];

  if(expected.size() != outputs->size()) {
    throw "Incorrect parameter size";
  }

  std::vector<double> retVal;
  for(int i = 0; i < outputs->size(); i++) {
    double err = .5 * pow(expected[i] - (*outputs)[i]->get_output(), 2);
    retVal.push_back(err);
  }

  return retVal;
}

void Net::to_s() {
  std::cout << "Inputs" << std::endl;
  for(int i = 0; i < input_count; i++) {
    std::cout << "\tInput" << i << ": " << this->inputs[i] << std::endl;
    for(int w = 0; w < weights_arr[0][i].size(); w++) {
      std::cout << "\t\tWeight" << w << ": " << weights_arr[0][i][w] << std::endl;
    }
  }

  //print input Bias
  std::cout << "\tBias" << std::endl;
  for(int i = 0; i < weights_arr[0].back().size(); i++) {
    std::cout << "\t\tWeight" << i << ": " << weights_arr[0].back()[i] << std::endl;
  }

  for(int i = 0; i < layers.size(); i++) {
    if(i == layers.size() - 1) {
      std::cout << "Output" << std::endl;
    }
    else {
      std::cout << "Layer " << i << std::endl;
    }
    for(int j = 0; j < layers[i].size(); j++) {
      std::cout << "\tPerceptron " << j << ", value: " << layers[i][j]->get_output() << ", grad: " << layers[i][j]->get_grad() << std::endl;
      if(i != layers.size() - 1) {
        for(int k = 0; k < weights_arr[i+1][j].size(); k++) {
          std::cout << "\t\tWeight" << k << " " << k << ", value: " << weights_arr[i+1][j][k] << std::endl;
        }
      }
    }
    if(i != layers.size() - 1) {
      std::cout << "\tBias " << std::endl;
      for(int b = 0; b < weights_arr[i+1].back().size(); b++) {
        std::cout << "\t\tWeight" << b << ": " << weights_arr[i].back()[b] << std::endl;
      }
    }
  }
}

void Net::train(TrainingData data ){
  for(int i = 0; i < data.size(); i++) {
    std::cout << "Training: " << i << std::endl;
    std::cout << "\tInputs: ";
    for(int j = 0; j < data[i].first.size(); j++) {
      std::cout << data[i].first[j] << " ";
    }
    std::vector<double> result = forward(data[i].first);

    std::cout << std::endl << "\tOutputs: ";
    for(int j = 0; j < data[i].second.size(); j++) {
      std::cout << result[j] << "," << data[i].second[j] << " ";
    }
    std::cout << std::endl;
    std::vector<double> err = get_error(data[i].second);
    std::cout << "\tErrors: ";
    for(int j = 0; j < err.size(); j++) {
      std::cout << err[j] << " ";
    }
    std::cout << std::endl << std::endl;
    back_prop(data[i].second);
  }
}
