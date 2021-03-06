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
#include <time.h>
#include <csignal>


#define DEF_TRTE_INTER 10
#define DEF_TARGET .2
#define DEF_MOMENTUM .5
#define DIVERGE_COUNT 3
#define DEF_INPUTS 11
#define DEF_TIMEOUT 450
#define DEF_OUTPUTS 4
#define RELU_STAT_FILE "relu.csv"
#define SIG_STAT_FILE "sig.csv"
#define TANH_STAT_FILE "tanh.csv"
#define ANALYTICS_DIR "Analytics/"
#define DEF_DEMO_FILE "Datasets/9demo_demo.dat"
#define DEF_KEYS_FILE "Keys/genres_clipped.key"

namespace po = boost::program_options;
using namespace net;

net::Net* demo_net;
bool is_demo;
bool is_training;

/*
* Checks if a specified file exists
* Code from user PherricOxide
* http://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c
*/
bool file_exists(const std::string filename) {
  struct stat buffer;
  return (stat (filename.c_str(), &buffer) == 0);
}

void signal_catcher(int sig) {
  if ((!is_training && is_demo) || !is_demo) {
    exit(0);
  }
  else {
    std::cout << "STOPPING TRAINING" << std::endl;
    demo_net->stop_training();
    is_training = false;
  }
}

std::vector<DataSet> convert_to_multi(DataSet orig_set, int feature_count) {
  std::vector<DataSet> retval(feature_count);
  arma::vec true_vec(2);
  true_vec(0) = 1;
  true_vec(1) = 0;
  arma::vec false_vec(2);
  false_vec(0) = 0;
  false_vec(1) = 1;

  for(auto t : orig_set) {
    for(int i = 0; i < feature_count; i++) {
      arma::vec temp_out((t.second[i] == 1) ? true_vec : false_vec);
      arma::vec temp_in(t.first);
      retval[i].push_back(std::make_pair(temp_in, temp_out));
    }
  }

  return retval;
}

DataSet read_file(const std::string filename, const int input_count, const int class_count, const int max_outputs=1) {
  DataSet sets;
  std::ifstream ff;
  try {
    ff.open(filename);
  }
  catch(std::exception e) {
    std::cerr << "Could not open file " << filename << ", Error: " << e.what() << std::endl;
    throw;
  }


  std::string line;
  while(std::getline(ff, line)) {
    std::stringstream ss;
    ss << line;
    std::vector<double> inputs;
    std::vector<double> outputs(class_count, 0);
    std::vector<int> expected_classes;
    double dtemp;
    int itemp;

    for(int i = 0; i < input_count; i++) {
      ss >> dtemp;
      inputs.push_back(dtemp);
    }

    int count = 0;
    while(count < max_outputs) {
      count++;
      ss >> itemp;
      expected_classes.push_back(itemp);
      if(!ss) {
        break;
      }
    }


    for(int i = 0; i < expected_classes.size(); i++) {
      outputs[expected_classes[i]] = 1;
    }

    sets.push_back(std::make_pair(arma::vec(inputs), arma::vec(outputs)));

  }
  ff.close();

  return sets;
}

std::vector<std::string> load_keys(std::string filename) {
  std::vector<std::string> retval;
  std::ifstream f(filename);
  std::string s;
  while(f >> s) {
    retval.push_back(s);
  }
  return retval;
}

DemoSet read_demo_file(std::string demo_file) {
  std::string line;
  std::ifstream f(demo_file);
  DemoSet retval;

  while(std::getline(f, line)) {
    std::stringstream s(line);
    std::vector<double> inputs;
    double val;
    char comma_catcher;
    for(int i = 0; i < 11; i++) {
      s >> val >> comma_catcher;
      inputs.push_back(val);
    } 
    std::string song_name;
    std::getline(s, song_name);
    retval.push_back(std::make_pair(arma::vec(inputs), song_name));
  }
  return retval;
}

int main(int argc, char** argv) {
  //Setup argument parsing
  po::options_description desc("Allowed Arguments");
  desc.add_options()
    ("help,h", "Display all arguments and their action")
    ("testfile", po::value<std::string>(), "The file to pull test data from")
    ("trainfile", po::value<std::string>(), "The file to pull training sets from")
    ("relu,", po::bool_switch()->default_value(false), "Use relu activation instead")
    ("multiclass", po::bool_switch()->default_value(false), "Use multiclass classification")
    ("multicount", po::value<int>(), "The count of classes to use in multiclass classification")
    ("target,t", po::value<double>(), "The target error to hit while training")
    ("tanh", po::bool_switch()->default_value(false), "Use TanH Activation")
    ("verbose,v", po::bool_switch()->default_value(false), "Print out more information during training and testing")
    ("interval,i", po::value<int>(), "The amount of training iterations before testing for error")
    ("inputcount", po::value<int>(), "The amount of inputs to the network (ONLY USED FOR BENCHMARKING)")
    ("outputcount", po::value<int>(), "The amount of outputs from the network (ONLY USED FOR BENCHMARKING)")
    ("momentum,m", po::value<double>(), "The value for momentum (0 <= m <= 1)")
    ("dropout", po::bool_switch()->default_value(false), "Use dropout in this neural net")
    ("timeout", po::value<int>(), "The timeout in seconds of training")
    ("diverge", po::value<int>(), "The count of consecutive divergence in the validation set before stopping training (also the upper bound in benchmarking)")
    ("benchmark", po::bool_switch()->default_value(false), "Test the different configurations and store the results in a CSV")
    ("dimensions,d", po::value<std::vector<int>>()->multitoken(), "The topology of the neural network (not including bias nodes, only used for individual neural nets)")
    ("demo", po::bool_switch()->default_value(false), "Setup the environment to demo");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  //Check for help
  if(vm.count("help")) {
    std::cout << desc << std::endl;
    std::exit(0);
  }

  std::signal(SIGINT, signal_catcher);

  //declare variables
  std::string testfile;
  std::string trainfile;
  int diverge_count = DIVERGE_COUNT;
  ClassificationType class_type = Single;
  int trte_inter = DEF_TRTE_INTER;
  double target = DEF_TARGET;
  int inputcount = DEF_INPUTS;
  int outputcount = DEF_OUTPUTS;
  int multicount = 1;
  int timeout = DEF_TIMEOUT;
  is_demo = vm["demo"].as<bool>();


  if(vm.count("multicount")) {
    multicount = vm["multicount"].as<int>();
  }

  if (vm.count("inputcount")) {
    inputcount = vm["inputcount"].as<int>();
  }

  if (vm.count("outputcount")) {
    outputcount = vm["outputcount"].as<int>();
  }

  if(vm.count("diverge")) {
    diverge_count = vm["diverge"].as<int>();
  }

  if(vm.count("timeout")) {
    timeout = vm["timeout"].as<int>();
  }

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


  if(vm.count("target")) {
    target = vm["target"].as<double>();
  }

  if(vm["multiclass"].as<bool>()) {
    class_type = Multi;
  }

  if(vm.count("interval")) {
    trte_inter = vm["interval"].as<int>();
  }

  if(vm["benchmark"].as<bool>()) {

    if(vm["verbose"].as<bool>()) {
      std::cout << "Reading training file" << std::endl;
    }
    DataSet training_sets = read_file(trainfile, inputcount, outputcount);
    //shuffle contents
    std::random_shuffle(training_sets.begin(), training_sets.end());
    if(vm["verbose"].as<bool>()) {
      std::cout << "Done Reading training file" << std::endl;
      std::cout << "Reading Testing file" << std::endl;
    }
    DataSet testing_sets = read_file(testfile, inputcount, outputcount);
    //shuffle contents
    std::random_shuffle(testing_sets.begin(), testing_sets.end());

    if(vm["verbose"].as<bool>()) {
      std::cout << "Done Reading testing file" << std::endl;
    }

    size_t extension_index = trainfile.find_last_of(".");
    size_t path_index = trainfile.find_last_of("/") + 1;
    std::string trainfile_noext = trainfile.substr(path_index, extension_index - path_index);

    std::cout << "Benchmarking..." << std::endl;
    for(int i = 0; i < 1; i++) {
      ActivationType act;
      std::string act_string = "";
      std::string filename = "";
      std::string connector;
      if (vm["dropout"].as<bool>()) {
        connector = "_D_";
      }
      else {
        connector = "_";
      }
       switch(i) {
        case 1:
          act = ReLU;
          filename = ANALYTICS_DIR + trainfile_noext + connector + RELU_STAT_FILE;
          act_string = "ReLU";
          break;
        case 0:
        act = Sigmoid;
        filename = ANALYTICS_DIR + trainfile_noext + connector + SIG_STAT_FILE;
        act_string = "Sigmoid";
        break;
        case 2:
          act = TanH;
          filename = ANALYTICS_DIR + trainfile_noext + connector + TANH_STAT_FILE;
          act_string = "TanH";
          break;
      }
      std::cout << "Print Analytics to " << filename << std::endl;
      std::ofstream of(filename);
      double max_accuracy = 0;
      std::string max_dimensions = "";
      double max_momentum = 0;
      of << "Dimensions,Momentum,Accuracy,Training Time\n";

      //h1 starts at the input plus 25% of the input
      for(int h1 = inputcount + (int)(inputcount * .25); h1 >= outputcount; h1--) {
        //inputs + outputs * .666, then subtract h1
        //If this is under 0, set to 0
        int h2_start = ((inputcount + outputcount) * (2.0/3.0)) - h1;
        h2_start = h2_start < 0 ? 0 : h2_start; 
        for(int h2 = h2_start; h2 >= 0; h2--) {
          if (h2 != 0 && h2 < h1) {
            continue;
          }
          std::vector<int> dimensions;
          std::string dimen_string = std::to_string(inputcount) + " " + std::to_string(h1) + " ";
          dimensions.push_back(inputcount);
          dimensions.push_back(h1);
          if(h2 != 0) {
            dimensions.push_back(h2);
            dimen_string += std::to_string(h2) + " ";
          }
          dimensions.push_back(outputcount);
          dimen_string += std::to_string(outputcount);
          for(double m = 0.1; m < .999; m += 0.1) {
            std::clock_t ts, te;
            Net net(dimensions, class_type, act, m, vm["verbose"].as<bool>(), vm["dropout"].as<bool>());

            ts = clock();
            net.train_and_test(training_sets, testing_sets, target, trte_inter, diverge_count, timeout);
            te = clock();

            float runtime = ((float)te - (float)ts);
            double acc = net.test(testing_sets);
            if (acc > max_accuracy) {
              max_accuracy = acc;
              max_dimensions = dimen_string;
              max_momentum = m;
            }

            of << dimen_string << "," << m  << "," << acc << "," << runtime << "\n";
            std::cout << "Tested " << act_string << ", Dimensions: [" << dimen_string
                      << "], Momentum: " << m << ", Result: "
                      << acc * 100 << "%, Training time: " << (runtime/CLOCKS_PER_SEC) << std::endl;
          }
          
        }
      }
      of.close();
      std::cout << std::endl << std::endl;
      std::cout << act_string << " Done Benchmarking, Best Setup: " << std::endl;
      std::cout << "\tDimensions: [" << max_dimensions << "], Momentum: "
                << max_momentum << ", Accuracy: "
                << max_accuracy << std::endl << std::endl << std::endl;
    }
  }
  else 
  {
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

    if(vm["verbose"].as<bool>()) {
        std::cerr << "Loading Training Datasets... " << std::endl;
    }
    DataSet training_sets = read_file(trainfile, dimensions[0], outputcount, multicount);
    if(vm["verbose"].as<bool>()) {
        std::cerr << "Done Loading Training Datasets, Shuffling... " << std::endl;
    }
    //shuffle contents
    std::random_shuffle(training_sets.begin(), training_sets.end());
    if(vm["verbose"].as<bool>()) {
        std::cerr << "Done Shuffling Training Datasets" << std::endl;
        std::cerr << "Loading Testing Datasets" << std::endl;
    }
    DataSet testing_sets = read_file(testfile, dimensions[0], outputcount, multicount);
    if(vm["verbose"].as<bool>()) {
      std::cerr << "Done Loading Testing datasets, Shuffling" << std::endl;
    }
    //shuffle contents
    std::random_shuffle(testing_sets.begin(), testing_sets.end());
    if(vm["verbose"].as<bool>()) {
      std::cerr << "All Datasets Loaded and Shuffled" << std::endl;
    }


    if (vm["multiclass"].as<bool>()) {

      std::vector<DataSet> training_sets_vec = convert_to_multi(training_sets, outputcount);

      std::vector<DataSet> testing_sets_vec = convert_to_multi(testing_sets, outputcount);

      std::vector<double> results(outputcount);
      std::vector<bool> is_correct_vec(testing_sets_vec[0].size(), true);

      std::vector<Net*> nn_vec(outputcount);
      for (int i = 0; i < nn_vec.size(); i++) {
        nn_vec[i] = new Net(dimensions, class_type, act_type, momentum, vm["verbose"].as<bool>(), vm["dropout"].as<bool>());
        nn_vec[i]->train_and_test(training_sets_vec[i], testing_sets_vec[i], target, trte_inter, diverge_count, timeout);
        double correct = 0;
        for(int j = 0; j < testing_sets_vec[i].size(); j++) {
          if(!nn_vec[i]->test_one(testing_sets_vec[i][j])) {
            is_correct_vec[j] = false;
          }
          else {
            correct++;
          }
        }
        results[i] = correct / testing_sets_vec[i].size();
        delete nn_vec[i];
      }

      double total_correct = 0;
      for(int c = 0; c < is_correct_vec.size(); c++) {
        if(is_correct_vec[c]) {
          total_correct++;
        }
      }

      std::cout << "Results:" << std::endl;
      for(int x = 0; x < results.size(); x++) {
        std::cout << "\t" << results[x] * 100 << std::endl;
      }

      std::cout << "Total correct: " << total_correct << "/" << is_correct_vec.size() << ", Total Accuracy: " << total_correct / is_correct_vec.size() * 100 << "%" << std::endl;;
    }
    else {

      if(vm["relu"].as<bool>() && vm["tanh"].as<bool>()) {
        std::cerr << "You cannot select both ReLU and TanH as your activation type" << std::endl;
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

      if(vm["verbose"].as<bool>()) {
        std::cerr << " Creating Network..." << std::endl;
      }
      demo_net = new Net(dimensions, class_type, act_type, momentum, vm["verbose"].as<bool>(), vm["dropout"].as<bool>());

      if(vm["verbose"].as<bool>()) {
        std::cerr << "Starting Training..." << std::endl;
      }

      std::clock_t ts, te;
      ts = clock();
      is_training = true;
      demo_net->train_and_test(training_sets, testing_sets, target, trte_inter, diverge_count, timeout);
      is_training = false;
      te = clock();

      double acc = demo_net->test(testing_sets);
      std::cout << "Finished, Accuracy: " << acc * 100 << "%, Train time: " << (te-ts) / (CLOCKS_PER_SEC + 0.0)<< std::endl;
      
      if(is_demo)  {
        std::cout << "Training Stopped" << std::endl << std::endl;

        DemoSet demo_sets = read_demo_file(DEF_DEMO_FILE);
        std::vector<std::string> keys = load_keys(DEF_KEYS_FILE);


        int choice;
        for(int i = 0; i < demo_sets.size(); i++) {
          std::cout << "\t(" << i << ") " + demo_sets[i].second << std::endl;
        }

        while(true) {
          choice = -1;
          while(choice < 0 || choice >= demo_sets.size()) {
            std::cout << "Enter a song number (0-" << demo_sets.size() - 1 << "): ";
            std::cin >> choice;
            if (choice < 0 || choice >= demo_sets.size()) {
              std::cout << choice << " is not an option!" << std::endl;
            }
          }
          arma::vec res = demo_net->forward_test(demo_sets[choice].first);
          int max_index = -1;
          double max_val = 0;
          for(int j = 0; j < res.size(); j++) {
            if(res(j) > max_val) {
              max_index = j;
              max_val = res(j);
            }
          }
          std::cout << "I think this is " << keys[max_index] << std::endl << std::endl;
        }
      }
      delete demo_net;
    }
  }

  exit(0);

}
