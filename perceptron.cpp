#include <iostream>
#include "perceptron.hpp"

using namespace net;

Perceptron::Perceptron() {
  this->output = 1;
}

Perceptron::~Perceptron() {

}

double Perceptron::get_grad() const {
  return this->grad;
}

double Perceptron::forward(const std::vector<double>& outputs, const std::vector<double>& weights) {
  if(outputs.size() != weights.size()) {
    std::cerr << "Parameter size mismatch: outputs: " << outputs.size() << ", weights: " << weights.size() << std::endl;
    throw ;
  }

  double sum = 0;
  for (int i = 0; i < outputs.size(); i++) {
    sum += outputs[i] * weights[i];
  }

  this->output = activation(sum);
  return this->output;
}

double Perceptron::backward() {
  // this->grad = this->output * (1 - this->output);
  // for(Edge* e : outputs) {
  //   this->grad += e->get_weight() * e->get_end()->get_grad();
  // }
  return -1;
}

double Perceptron::activation(const double x) const {
  return -1;
}

double Perceptron::get_output() const{
  return this->output;
}

void Perceptron::set_grad(const double grad) {
  this->grad = grad;
}

void Perceptron::set_output(const double new_output) {
  this->output = new_output;
}
