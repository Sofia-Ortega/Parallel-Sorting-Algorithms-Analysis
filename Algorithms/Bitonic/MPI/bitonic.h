
/**
 * -------------------- Adapted From -----------------------------------
 * Code: https://github.com/adrianlee/mpi-bitonic-sort/tree/master
 * Report: https://andreask.cs.illinois.edu/Teaching/HPCFall2012/Projects/yourii-report.pdf
 * Author: Adrian Lee
 * University: McGill University
 * Date: Dec 1, 2012
 * 
 *     References:
        Assignment 4 Supplement - Parallel Bitonic Sort Algorithm
        Sorting Algorithms - http://www-users.cs.umn.edu/~karypis/parbook/Algorithms/pchap9.pdf
        Bionic Sorting - http://www.thi.informatik.uni-frankfurt.de/~klauck/PDA07/Bitonic%20Sorting.pdf
*/

void SequentialSort(void);
void CompareLow(int bit);
void CompareHigh(int bit);
int ComparisonFunc(const void * a, const void * b);