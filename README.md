# Parallel Computing (CSCE435) Project

This project evaluates the performance of four parallel sorting algorithms across different input types, levels of parallelization, and parallelization frameworks. Each algorithm was tested with four input types (random, reversed, sorted, and perturbed) and implemented using both CUDA and MPI to compare their efficiency and scalability.

Go to [Report.md](./Report.md) to find final **data analysis**

### Algorithms:
- Bitonic Sort
- Merge Sort
- Quick Sort
- Radix Sort

### Input Types:
- Random
- Reversed
- Sorted
- Perturbed (sorted data with slight modifications)

### Parallelization Frameworks
- CUDA
- MPI 


## Data
All performance profiling data files can be found in the `PerformanceEval/dataFiles` directory as Caliper Trace files. 

## Data Analysis
Go to [Report.md](./Report.md) to find final **data analysis**

In addition, this is a link to a [slideshow](https://docs.google.com/presentation/d/1xVNeRjE2JDYVoHVwIL6yXYmSHAhrraA6gHH12zjqB2k/edit?usp=sharing) where we presented our findings

## Jupyter Notebooks
The jupyter notebooks used to generate our plots are as follows: 

- `PerformanceEval/AlgorithmComparisons.ipynb` and `PerformanceEval/WeakScalingComp.ipynb` are the notebooks we used to compare all of our algorithms together 
- `PerformanceEval/BitonicSortPlotting.ipynb` is the notebook we used to plot results for our Bitonic Sort Algorithm
- `PerformanceEval/MergeSortPlotting.ipynb` is the notebook we used to plot results for our Merge Sort Algorithm
- `PerformanceEval/RadixSortPlotting.ipynb` is the notebook we used to plot results for our Radix Sort Algorithm
- `PerformanceEval/QuickSortPlotting.ipynb` is the notebook we used to plot results for our Quick Sort Algorithm


In addition, the slideshow displaying the comparisons between the algorithms can be found [here](https://docs.google.com/presentation/d/1xVNeRjE2JDYVoHVwIL6yXYmSHAhrraA6gHH12zjqB2k/edit?usp=sharing)

## Contributors
Below is the work of the main contributors and the files they mainly worked on 

Sofia Ortega:
- Radix Sort Implementation (`Algorithms/radix/`)
- Algorithm Comparisions for Strong Scaling (`PerformanceEval/AlgorithmComparisons.ipynb`)
- Algorithm Comparison for Speedup (`PerformanceEval/AlgorithmComparisons.ipynb`)

Will Thompson: 
- Merge Sort Implementation (`Algorithms/MergeSort/`)
- Algorithm Comparisons for Weak Scaling (`PerformanceEval/WeakScalingComp.ipynb`)


Krithivel Ramesh: 
- Bitonic Sort Implementation (`Algorithms/Bitonic/`)

Sriabhinandan Venkataraman: 
- Quick Sort Implementation (`Algorithms/quicksort/`)
