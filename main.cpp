#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <tuple>
#include "boost/program_options.hpp"
#include "net.hpp"

namespace po = boost::program_options;
using namespace net;

/*
* Checks if a specified file exists
* Code retrieved from user PherricOxide
* http://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c
*/
bool file_exists(std::string filename) {
  struct stat buffer;
  return (stat (filename.c_str(), &buffer) == 0);
}

int main(int argc, char** argv) {
  //Setup argument parsing
  po::options_description desc("Allowed Arguments");
  desc.add_options()
    ("help", "Display all arguments and their action")
    ("testfile", po::value<std::string>(), "The file to pull test data from")
    ("trainfile", po::value<std::string>(), "The file to pull training sets from")
    ("traincount", po::value<int>(), "The amount of times to iterate over the train data (default 1)")
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
  int traincount = 1;
  std::vector<int> dimensions;

  //retrieve necessary parameters from args or user input
  if(vm.count("testfile")) {
    testfile = vm["testfile"].as<std::string>();
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

  if(vm.count("traincount")) {
    traincount = vm["traincount"].as<int>();
  }

  std::cout << std::endl << std::endl;

  std::cout << "~~ Neural Network ~~" << std::endl;
  std::cout << "Dimensions (excluding bias): ";
  for(int i = 0; i < dimensions.size(); i++) {
    std::cout << dimensions[i] < " ";
  }
  std::cout << std::endl;
  std::cout << "Train File: " << trainfile << std::endl;
  std::cout << "Test File: " << testfile << std::endl;
  std::cout << "Training Iterations: " << traincount << std::endl;
  std::cout << std::endl << std::endl;

  //print arguments

  TrainingData training_sets;
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 2; j++) {
      training_sets.push_back(std::make_pair(std::vector<double>{(double)i, (double)j}, std::vector<double>{(double)(i | j)}));
    }
  }

  Net net(dimensions, Sigmoid);
  for(int i = 0; i < traincount; i++) {
    net.train(training_sets);
  }
  net.to_s();
}
