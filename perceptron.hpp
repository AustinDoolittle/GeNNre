#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>

namespace net {
  class Perceptron{
  protected:
    double output;
    double grad;
    virtual double activation(const double x) const;
  public:
    Perceptron();
    virtual ~Perceptron();
    double forward(const std::vector<double>& outputs, const std::vector<double>& weights);
    double backward();
    double get_output() const;
    void set_grad(const double grad);
  };

  class SigmoidPerceptron: public Perceptron{
  protected:
    virtual double activation(const double x) const;
  public:
    SigmoidPerceptron();
    ~SigmoidPerceptron();
  };

  class ReLUPerceptron: public Perceptron{
  protected:
    virtual double activation(const double x) const;
  public:
    ReLUPerceptron();
    ~ReLUPerceptron();
  };
}

#endif
