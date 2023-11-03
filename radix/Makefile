EXECUTABLES = dataset_gen radix_sort p_radix_sort
DEBUG_FLAGS = -O3

all: $(EXECUTABLES)

dataset_gen: dataset_gen.c
	gcc -std=gnu99 $(DEBUG_FLAGS) -o$@ $^ -lrt

radix_sort: radix_sort.c
	gcc -std=gnu99 $(DEBUG_FLAGS) -o$@ $^ -lrt

p_radix_sort: p_radix_sort.c
	mpicc -std=gnu99 $(DEBUG_FLAGS) -o$@ $^ -lrt

%.o : %.c %.h
	mpicc $(DEBUG_FLAGS) -c -std=gnu99 $<

clean:
	rm -f $(EXECUTABLES) *.o mpe-*