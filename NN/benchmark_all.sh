#!/bin/sh

dataset_dir="Datasets"
flags="--target 0.001  --interval 50 --benchmark --inputcount 11 --timeout 1500 "

#build
make clean
make

#run 2 class benchmark, no dropout
#./NN.out --trainfile $dataset_dir/2train_norm_data.dat --testfile $dataset_dir/2test_norm_pca_data.dat $flags --outputcount 2 --diverge 5 

#run 2 class benchmark, dropout
#./NN.out --trainfile $dataset_dir/2train_norm_data.dat --testfile $dataset_dir/2test_norm_pca_data.dat $flags --outputcount 2 --diverge 5 --dropout 

#run 5 class benchmark, no droupout
#./NN.out --trainfile $dataset_dir/5train_norm_data.dat --testfile $dataset_dir/5test_norm_pca_data.dat $flags --outputcount 5 --diverge 4 

#run 5 class benchmark, dropout
#./NN.out --trainfile $dataset_dir/5train_norm_data.dat --testfile $dataset_dir/5test_norm_pca_data.dat $flags --outputcount 5 --diverge 4 --dropout 

#run 12 class benchmark, no dropout
./NN.out --trainfile $dataset_dir/9train_norm_data.dat --testfile $dataset_dir/9test_norm_pca_data.dat $flags --outputcount 9 --diverge 8

#run 12 class benchmarck, dropout
#./NN.out --trainfile $dataset_dir/9train_norm_data.dat --testfile $dataset_dir/9test_norm_pca_data.dat $flags --outputcount 9 --diverge 4 --dropout 

wait

echo all done!

exit 0
