#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <tuple>
#include <time.h>
#include <algorithm>
#include "boost/program_options.hpp"
#include "net.hpp"

#define DEF_TRTE_INTER 10
#define DEF_TARGET .2
#define DEF_MOMENTUM .5
#define DEF_INPUTS 11
#define DEF_OUTPUTS 2
#define RELU_STAT_FILE "relu_stat.csv"
#define SIG_STAT_FILE "sig_stat.csv"
#define TANH_STAT_FILE "tanh_stat.csv";

namespace po = boost::program_options;
using namespace net;

/*
* Checks if a specified file exists
* Code retrieved from user PherricOxide
* http://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c
*/
bool file_exists(const std::string filename) {
  struct stat buffer;
  return (stat (filename.c_str(), &buffer) == 0);
}

DataSet read_file(const std::string filename, const int input_count, const int class_count, const int max_outputs=1) {
  DataSet sets;
  std::ifstream ff;
  ff.open(filename);
  std::string line;
  while(std::getline(ff, line)) {
    std::stringstream ss;
    ss << line;
    std::vector<double> inputs;
    std::vector<double> outputs;
    std::vector<int> expected_classes;
    double dtemp;
    int itemp;
    for(int i = 0; i < input_count; i++) {
      ss >> dtemp;
      inputs.push_back(dtemp);
    }

    for(int i = 0; i < max_outputs; i++) {
      ss >> itemp;
      if(ss.fail()) {
        break;
      }
      else {
        expected_classes.push_back(itemp);
      }
    }

    for(int i = 0; i < class_count; i++) {
      if(std::find(expected_classes.begin(), expected_classes.end(), i) != expected_classes.end()) {
        outputs.push_back(1);
      }
      else {
        outputs.push_back(0);
      }
    }

    sets.push_back(std::make_pair(arma::vec(inputs), arma::vec(outputs)));
  }
  ff.close();
  return sets;
}

int main(int argc, char** argv) {
  //Setup argument parsing
  po::options_description desc("Allowed Arguments");
  desc.add_options()
    ("help", "Display all arguments and their action")
    ("testfile", po::value<std::string>(), "The file to pull test data from")
    ("trainfile", po::value<std::string>(), "The file to pull training sets from")
    ("trainrate", po::value<double>(), "The weight to apply during back propogation")
    ("relu", po::bool_switch()->default_value(false), "Use relu activation instead")
    ("multiclass", po::bool_switch()->default_value(false), "Use multiclass classification")
    ("target", po::value<double>(), "The target error to hit while training")
    ("tanh", po::bool_switch()->default_value(false), "Use TanH Activation")
    ("verbose", po::bool_switch()->default_value(false), "Print out more information during training and testing")
    ("interval", po::value<int>(), "The amount of training iterations before testing for error")
    ("momentum", po::value<double>(), "The value for momentum (0 <= m <= 1)")
    ("benchmark", po::bool_switch()->default_value(false), "Test the different configurations and store the results in a CSV")
    ("dimensions", po::value<std::vector<int>>()->multitoken(), "The topology of the neural network (not including bias nodes)");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  //Check for help
  if(vm.count("help")) {
    std::cout << desc << std::endl;
    std::exit(0);
  }

  //declare variables
  std::string testfile;
  std::string trainfile;
  ClassificationType class_type = Single;
  int trte_inter = DEF_TRTE_INTER;
  double target = DEF_TARGET;

  //retrieve necessary parameters from args or user input
  if(vm.count("testfile")) {
    testfile = vm["testfile"].as<std::string>();
    if(!file_exists(testfile)) {
      std::cout << "Specified test file does not exist" << std::endl;
      exit(1);
    }
  }
  else {
    //testfile was not given in arguments, prompt user for file
    do {
      std::cout << "Please enter the testfile: ";
      std::cin >> testfile;
      if(file_exists(testfile)) {
        break;
      }
      else {
        std::cout << "File does not exist" << std::endl;
      }
    }
    while (true);
  }

  if(vm.count("trainfile")) {
    trainfile = vm["trainfile"].as<std::string>();
    if(!file_exists(trainfile)) {
      std::cout << "Specified train file does not exist" << std::endl;
      exit(1);
    }
  }
  else {
    //trainfile was not given in arguments, prompt user for file
    do {
      std::cout << "Please enter the trainfile: ";
      std::cin >> testfile;
      if(file_exists(testfile)) {
        break;
      }
      else {
        std::cout << "File does not exist" << std::endl;
      }
    }
    while (true);
  }

  DataSet training_sets = read_file(trainfile, DEF_INPUTS, DEF_OUTPUTS);
  //shuffle contents
  std::random_shuffle(training_sets.begin(), training_sets.end());

  DataSet testing_sets = read_file(testfile, DEF_INPUTS, DEF_OUTPUTS);
  //shuffle contents
  std::random_shuffle(testing_sets.begin(), testing_sets.end());

  if(vm.count("target")) {
    target = vm["target"].as<double>();
  }

  if(vm["multiclass"].as<bool>()) {
    class_type = Multi;
  }

  if(vm.count("trteinter")) {
    trte_inter = vm["trteinter"].as<int>();
  }

  if(vm["benchmark"].as<bool>()) {
    std::cout << "Benchmarking..." << std::endl;
    for(int i = 0; i < 3; i++) {
      ActivationType act;
      std::string act_string = "";
      std::string filename = "";
      switch(i) {
        case 0:
          act = Sigmoid;
          filename = SIG_STAT_FILE;
          act_string = "Sigmoid";
          break;
        case 1:
          act = ReLU;
          filename = RELU_STAT_FILE;
          act_string = "ReLU";
          break;
        case 2:
          act = TanH;
          filename = TANH_STAT_FILE;
          act_string = "TanH";
          break;
      }
      std::ofstream of(filename);
      of << ",0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9\n";
      for(int h1 = 1; h1 < DEF_INPUTS + 2; h1++) {
        for(int h2 = 0; h2 < DEF_INPUTS + 2; h2++) {
          std::vector<int> dimensions;
          std::string dimen_string = std::to_string(DEF_INPUTS) + " " + std::to_string(h1) + " ";
          dimensions.push_back(DEF_INPUTS);
          dimensions.push_back(h1);
          if(h2 != 0) {
            dimensions.push_back(h2);
            dimen_string += std::to_string(h2) + " ";
          }
          dimensions.push_back(DEF_OUTPUTS);
          dimen_string += std::to_string(DEF_OUTPUTS);
          of << dimen_string;
          for(double m = 0.1; m < .999; m += 0.1) {
            std::clock_t ts, te;
            Net net(dimensions, class_type, act, m, vm["verbose"].as<bool>());

            ts = clock();
            net.train_and_test(training_sets, testing_sets, target, trte_inter);
            te = clock();

            float runtime = ((float)te - (float)ts);
            double acc = net.test(testing_sets);

            of << "," << acc;
            std::cout << "Tested " << act_string << ", Dimensions: [" << dimen_string
                      << "], Momentum: " << m << ", Result: " << acc * 100 << "%, Training time: " << runtime << std::endl;
          }
          of << "\n";
        }
      }
      of.close();
    }
  }
  else {
    double momentum = DEF_MOMENTUM;
    ActivationType act_type = Sigmoid;
    std::vector<int> dimensions;

    if(vm.count("momentum")) {
      momentum = vm["momentum"].as<double>();
    }

    if(vm.count("dimensions")) {
      dimensions = vm["dimensions"].as<std::vector<int>>();
    }
    else {
        std::cout << "You must specify the dimensions in the arguments" << std::endl;
        exit(1);
    }

    if(vm["relu"].as<bool>() && vm["tanh"].as<bool>()) {
      std::cerr << "You cannot select both ReLU and Softplus as your activation type" << std::endl;
      exit(2);
    }
    else if (vm["relu"].as<bool>()) {
      act_type = ReLU;
    }
    else if(vm["tanh"].as<bool>()) {
      act_type = TanH;
    }

    std::cout << std::endl << std::endl;

    std::cout << "~~ Neural Network ~~" << std::endl;
    std::cout << "Dimensions (excluding bias): ";
    for(int i = 0; i < dimensions.size(); i++) {
      std::cout << dimensions[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Train File: " << trainfile << std::endl;
    std::cout << "Test File: " << testfile << std::endl;
    std::cout << std::endl << std::endl;

    Net net(dimensions, class_type, act_type, momentum, vm["verbose"].as<bool>());

    net.train_and_test(training_sets, testing_sets, target, trte_inter);

    double acc = net.test(testing_sets);
    std::cout << "Finished, Accuracy: " << acc * 100 << "%" << std::endl;
  }



  exit(0);

}
