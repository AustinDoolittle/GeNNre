FLAGS=-std=c++11 -Wall
CXX=g++
OUT=nn

SRC=$(wildcard *.cpp)
OBJ=$(SRC:%.cpp=%.o)

all: $(OBJ)
	$(CXX) $(FLAGS) -o $(OUT) $^

%.o: %.cpp
	$(CXX) $(FLAGS) $< -c -o $@

clean:
	rm -f *.o
	rm $(OUT)
