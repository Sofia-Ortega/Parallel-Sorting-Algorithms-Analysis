# Compiler and Flags
CXX = g++
CXXFLAGS = -std=c++11
LDFLAGS = -lm -fopenmp

# Executable name
EXEC = quicksort_mpi

# Source and Object files
SRC = quicksort_mpi.cpp
OBJ = $(SRC:.cpp=.o)

# MPI Compiler
MPICXX = mpicxx

all: $(EXEC)

$(EXEC): $(OBJ)
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(EXEC)
