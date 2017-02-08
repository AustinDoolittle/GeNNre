#include "net.hpp"

using namespace net;

Net::Net(std::vector<int> dimensions, ActivationType type) {
  this->activation_type = type;
  this->input_count = dimensions[0];

  for(int i = 1; i < dimensions.size(); i++) {
    Weights w(dimensions[i-1], std::vector<double>(dimensions[i]));
    Layer l;
    switch(type) {
      case Sigmoid:
        l = std::make_pair(w, std::vector<Perceptron>(dimensions[i], SigmoidPerceptron()));
        break;
      case ReLU:
        //TODO
        break;
    }
    this->layers.push_back(l);
  }
}

Net::~Net() {

}
