import sys
import os

def normalize(x):
  return (x*0.001)/(0.001*16)

f = open('digits-test.dat', 'r')

lines = [x.strip() for x in f.readlines()]
for line in lines:
  words = line.split()
  new_line = []
  for i in range(0:64):
    new_line.append(normalize(int(words)))
