#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>


namespace net {
  class Perceptron{
  protected:
    double output;
    double grad;
    virtual double activation(double x);
  public:
    Perceptron();
    ~Perceptron();
    void forward();
    void backward();
  };

  class SigmoidPerceptron: public Perceptron{
  protected:
    double activation(double x);
  public:
    SigmoidPerceptron();
    ~SigmoidPerceptron();
  };

  class ReLUPerceptron: public Perceptron{
  protected:
    double activation(double x);
  public:
    ReLUPerceptron();
    ~ReLUPerceptron();
  };
}

#endif
