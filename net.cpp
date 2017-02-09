#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include "net.hpp"

using namespace net;

Net::Net(std::vector<int> dimensions, ActivationType type) {
  srand(time(NULL));
  this->activation_type = type;
  this->input_count = dimensions[0];

  for(int i = 1; i < dimensions.size(); i++) {
    // Weights w(dimensions[i], std::vector<double>(dimensions[i-1], (rand() % 3 - 1)));
    Weights w(dimensions[i], std::vector<double>(dimensions[i-1], 1));

    std::vector<Perceptron*> temp;

    for(int j = 0; j < dimensions[i]; j++) {
      if(type == Sigmoid) {
        temp.push_back(new SigmoidPerceptron());
      }
      else if(type == ReLU) {
        temp.push_back(new ReLUPerceptron());
      }
    }

    Layer l = std::make_pair(w, temp);

    this->layers.push_back(l);
  }
}

Net::~Net() {
  for(int i = 0; i < layers.size(); i++) {
    for(int j = 0; j < layers[i].second.size(); j++) {
      delete layers[i].second[j];
    }
  }
}


std::vector<double> Net::forward(const std::vector<double> inputs) {
  if (inputs.size() != this->input_count) {
    throw "Incorrect inputs size";
  }

  this->curr_inputs = inputs;

  std::vector<double> v1(inputs.begin(), inputs.end());
  std::vector<double> v2;
  std::vector<double>* prev = &v1;
  std::vector<double>* curr = &v2;

  for(int i = 0; i < layers.size(); i++) {
    for(int j = 0; j < layers[i].second.size(); j++) {
      double res = layers[i].second[j]->forward(*prev, layers[i].first[j]);
      curr->push_back(res);
    }

    std::vector<double>* temp = prev;

    prev = curr;
    curr = temp;
    curr->clear();
  }

  return *prev;
}

void Net::back_prop(const std::vector<double> expected) {

  std::vector<Perceptron*>* outputs = &layers[layers.size() - 1].second;
  if(expected.size() != outputs->size()) {
    throw "Incorrect parameter size";
  }

  //calculate last row error first
  for(int i = 0; i < outputs->size(); i++) {
    double val = (*outputs)[i]->get_output();
    double grad = val * (1 - val) * (expected[i] - val);
    (*outputs)[i]->set_grad(grad);
  }

  //calculate hidden
}

std::vector<double> Net::get_error(const std::vector<double> expected){

  std::vector<Perceptron*>* outputs = &layers[layers.size() - 1].second;

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
