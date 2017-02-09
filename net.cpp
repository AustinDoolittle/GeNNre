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

  for(int i = 0; i < dimensions.size(); i++) {
    int max = (i == (dimensions.size() - 1)) ? dimensions[i] : dimensions[i] + 1;


    if (i != 0) {
      // Weights w(dimensions[i-1], std::vector<double>(dimensions[i], 1));
      Weights w(dimensions[i-1] + 1, std::vector<double>(max, (rand() % 3 - 1)));

      this->weights_arr.push_back(w);
    }

    Layer temp;
    for(int j = 0; j < max; j++) {
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
  return retVal;
}

std::vector<double> Net::forward(const std::vector<double> inputs) {
  this->load_inputs(inputs);

  std::vector<double> v1 = get_outputs(0);

  for(int i = 1; i < layers.size(); i++) {
    Weights* w = &weights_arr[i - 1];

    int max = (i == (layers.size() - 1)) ? layers[i].size() : layers[i].size() - 1;
    for(int j = 0; j < max; j++) {
      std::vector<double> edges;
      for(int k = 0; k < (*w).size(); k++) {
        edges.push_back((*w)[k][j]);
      }

      layers[i][j]->forward(v1, edges);
    }

    v1 = get_outputs(i);
  }

  return get_outputs(layers.size()-1);
}

void Net::load_inputs(const std::vector<double> inputs) {
  if (inputs.size() != this->layers[0].size() - 1) {
    std::cerr <<  "Incorrect inputs size" << std::endl;
    throw;
  }

  for(int i = 0; i < inputs.size() - 1; i++) {
    layers[0][i]->set_output(inputs[i]);
  }
}

void Net::back_prop(const std::vector<double> expected) {

  Layer* outputs = &layers[layers.size() - 1];
  if(expected.size() != outputs->size()) {
    throw "Incorrect parameter size";
  }

  //calculate last row error first
  for(int i = 0; i < outputs->size(); i++) {
    double val = (*outputs)[i]->get_output();
    double grad = val * (1 - val) * (expected[i] - val);
    (*outputs)[i]->set_grad(grad);
  }

  //calculate hidden layers
  for(int i = layers.size() - 2; i >= 0; i--) {
    //calculate error on these perceptrons
    for(int j = 0; j < layers[i].size(); j++) {
      double grad = layers[i][j]->get_output() * (1 - layers[i][j]->get_output());
      for(int k = 0; k < weights_arr[i][j].size(); k++) {
        grad *= (weights_arr[i][j][k] * layers[i+1][k]->get_grad());
        weights_arr[i][j][k] += (this->train_rate * layers[i+1][k]->get_grad() * layers[i][j]->get_output());
      }
      if(j != (layers[i].size() - 1))
      {
        layers[i][j]->set_output(grad);
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
  for(int i = 0; i < layers.size(); i++) {
    std::cout << "Layer " << i << std::endl;
    for(int j = 0; j < layers[i].size(); j++) {
      std::cout << "\tPerceptron " << j << ", value: " << layers[i][j]->get_output() << std::endl;
      if(i != layers.size() - 1) {
        for(int k = 0; k < weights_arr[i][j].size(); k++) {
          std::cout << "\t\tWeight" << k << " " << k << ", value: " << weights_arr[i][j][k] << std::endl;
        }
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
