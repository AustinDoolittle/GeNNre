FLAGS=-std=c++11 -Wall
CXX=g++
OUT=nn.out
LIBS=-lboost_program_options

SRC=$(wildcard *.cpp)
OBJ=$(SRC:%.cpp=%.o)

all: $(OBJ)
	$(CXX) $(FLAGS) $(LIBS) -o $(OUT) $^

%.o: %.cpp
	$(CXX) $(FLAGS) $< -c -o $@

clean:
	rm -f *.o
	rm $(OUT)
