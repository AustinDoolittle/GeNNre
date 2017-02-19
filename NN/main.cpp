#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <tuple>
#include "boost/program_options.hpp"
#include "net.hpp"

#define DEF_TRAIN_RATE 0.5

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
    ("traincount", po::value<int>(), "The amount of times to iterate over the training data")
    ("verbose", po::bool_switch()->default_value(false), "Print out more information during training and testing")
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
  std::vector<int> dimensions;
  double trainrate = DEF_TRAIN_RATE;
  int traincount = 1;
  ActivationType act_type = Sigmoid;
  ClassificationType class_type = Single;

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

  if(vm.count("dimensions")) {
    dimensions = vm["dimensions"].as<std::vector<int>>();
  }
  else {
      std::cout << "You must specify the dimensions in the arguments" << std::endl;
      exit(1);
  }

  if(vm.count("trainrate")) {
    trainrate = vm["trainrate"].as<double>();
  }

  if(vm["relu"].as<bool>()) {
    act_type = ReLU;
  }

  if(vm["multiclass"].as<bool>()) {
    class_type = Multi;
  }

  if(vm.count("traincount")) {
    traincount = vm["traincount"].as<int>();
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


  DataSet training_sets = read_file(trainfile, dimensions[0], dimensions.back());

  Net net(dimensions, class_type, act_type, trainrate, vm["verbose"].as<bool>());
  std::cout << net.to_s() << std::endl;
  for(int i = 0; i < traincount; i++) {
    std::cout << i + 1 << "/" << traincount << " Training" << std::endl;
    net.train(training_sets);
  }
  std::cout << std::endl << "Final state:" << std::endl;
  net.to_s();

  DataSet testing_sets = read_file(testfile, dimensions[0], dimensions.back());

  std::cout << "Testing: " << std::endl;
  net.test(training_sets);

  exit(0);

}
