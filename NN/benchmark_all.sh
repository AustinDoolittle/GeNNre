#!/bin/sh

#run 2 class benchmark, no dropout
./NN.out --trainfile 2train_norm_data.dat --testfile 2test_norm_data.dat --target 0.001  --interval 10 --benchmark --inputcount 11 --outputcount 2 --diverge 6

#run 2 class benchmark, dropout
./NN.out --trainfile 2train_norm_data.dat --testfile 2test_norm_data.dat --target 0.001  --interval 10 --benchmark --inputcount 11 --outputcount 2 --diverge 6 --dropout

#run 5 class benchmark, no droupout
./NN.out --trainfile 5train_norm_data.dat --testfile 5test_norm_data.dat --target 0.001  --interval 10 --benchmark --inputcount 11 --outputcount 5 --diverge 4

#run 5 class benchmark, dropout
./NN.out --trainfile 5train_norm_data.dat --testfile 5test_norm_data.dat --target 0.001  --interval 10 --benchmark --inputcount 11 --outputcount 5 --diverge 4 --dropout

#run 12 class benchmark, no dropout
./NN.out --trainfile 12train_norm_data.dat --testfile 12test_norm_data.dat --target 0.001  --interval 10 --benchmark --inputcount 11 --outputcount 12 --diverge 4

#run 12 class benchmarck, dropout
./NN.out --trainfile 12train_norm_data.dat --testfile 12test_norm_data.dat --target 0.001  --interval 100 --benchmark --inputcount 11 --outputcount 12 --diverge 4 --dropout
