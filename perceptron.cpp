#include <math.h>
#include "perceptron.hpp"

using namespace net;

Perceptron::Perceptron() {

}

Perceptron::~Perceptron() {
}

void Perceptron::forward() {
  // double sum = 0;
  // for(Edge* e : inputs) {
  //   sum += e->get_weight() * e->get_front()->get_output();
  // }
  //this->output = this->activation(sum);
}
void Perceptron::backward() {
  // this->grad = this->output * (1 - this->output);
  // for(Edge* e : outputs) {
  //   this->grad += e->get_weight() * e->get_end()->get_grad();
  // }
}

double Perceptron::activation(double x) {
  return -1;
}

double SigmoidPerceptron::activation(double x) {
  return 1/(1+exp(-x));
}

SigmoidPerceptron::SigmoidPerceptron():Perceptron() {

}

SigmoidPerceptron::~SigmoidPerceptron() {

}

ReLUPerceptron::ReLUPerceptron():Perceptron() {

}

ReLUPerceptron::~ReLUPerceptron() {

}

double ReLUPerceptron::activation(double x) {
  //TODO
  return -1;
}
