import argparse as ap
import numpy as np

if __name__ == "__main__":
  parser = ap.ArgumentParser(description="This is a neural network written in python using the numpy library for matrix operations")
  parser.add_argument("--data", required=True, help="The datafile to pull from")
  parser.add_argument("-f", "--features", type=int, required=True, help="The number of input features")
  parser.add_argument("-k", "--classes", type=int, required=True, help="The number of classes")
  args = parser.parse_args()

  f = open(args.data, 'r')

  lines = f.readlines()

  f.close()
  input_arr = []
  output_arr = []
  for line in lines:
    words = line.split()

    inputs = []
    for feat in words[:-1]:
      inputs.append(float(feat))

    if len(inputs) != args.features:
      raise ValueError("The Dataset is formatted incorrectly. Expected " + str(args.features) + " features, got " + str(len(inputs)))

    outputs = int(words[args.features])

    input_arr.append(np.array(inputs))
    output_arr.append(np.array(outputs))

  retval = zip(input_arr, output_arr)

  s1 = int(len(retval) *.75)

  train_filename = str(args.classes) + "train_norm_pca_data.dat"
  test_filename = str(args.classes) + "test_norm_pca_data.dat"
  train_set = retval[:s1]
  test_set = retval[s1:]

  f_out = open(train_filename, 'w')
  for line in train_set:
    for num in line[0]:
      f_out.write(str(num) + " ")

    f_out.write(str(line[1]) + "\n")
  f_out.close()

  f_out = open(test_filename, 'w')
  for line in test_set:
    for num in line[0]:
      f_out.write(str(num) + " ")

    f_out.write(str(line[1]) + "\n")
  f_out.close()

