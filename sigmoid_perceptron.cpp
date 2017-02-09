#include <math.h>
#include <iostream>

#include "perceptron.hpp"

using namespace net;

double SigmoidPerceptron::activation(const double x) const {
  return 1.0/(1+exp(-x));
}

SigmoidPerceptron::SigmoidPerceptron():Perceptron() {

}

SigmoidPerceptron::~SigmoidPerceptron() {

}
