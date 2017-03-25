import sys
import os

def normalize(x):
  return x/16.0

f = open('digits-test.dat', 'r')

lines = [x.strip() for x in f.readlines()]
f.close()

f = open('digits-test-norm.dat', 'w')

for line in lines:
  words = line.split()
  for i in range(0,64):
    f.write(str(normalize(int(words[i]))) + " ")

  f.write(words[64] + "\n")

f.close()

