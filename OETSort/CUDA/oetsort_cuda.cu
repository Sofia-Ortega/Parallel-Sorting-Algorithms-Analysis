/**
 * -------------------- SOURCE -----------------------------------
 * Code: https://github.com/Kshitij421/Odd-Even-Sort-using-Cuda-/blob/master/oddeven.cu
 * Author: @Kshitij421
 * Date: March 16, 2016 
 * 
*/

#include<stdio.h>
#include<cuda.h>

const char* main = "main";
const char * genValuesTime = "data_init";
const char * correctness = "correctness_check";
const char * compSmall = "comp_small";
const char * compLarge = "comp_large";
const char * comm = "comm";
const char * commSmall = "comm_small";
const char * commLarge = "comm_large_h2d";
const char * commLarge = "comm_large_d2h";

__global__ void oddeven(int* x,int I,int n)
{
	CALI_CXX_MARK_FUNCTION;

	CALI_MARK_BEGIN("comp_small");
	
	int id=blockIdx.x;
	if(I==0 && ((id*2+1)< n)){
		if(x[id*2]>x[id*2+1]){
			int X=x[id*2];
			x[id*2]=x[id*2+1];
			x[id*2+1]=X;
		}
	}
	if(I==1 && ((id*2+2)< n)){
		if(x[id*2+1]>x[id*2+2]){
			int X=x[id*2+1];
			x[id*2+1]=x[id*2+2];
			x[id*2+2]=X;
		}
	}

	CALI_MARK_END("comp_small")
}

int main()
{
	CALI_CXX_MARK_FUNCTION;
	CALI_MARK_BEGIN("main");

	CALI_MARK_BEGIN("data_init");

	int a[100],n,c[100],i;
	int *d;

	printf("Enter how many elements of first array:");
	scanf("%d",&n);
	
	// geberate random values
	for(i=0; i<n; i++)
	{
		a[i] = rand()%100;
	}

	CALI_MARK_END("data_int");

	CALI_MARK_BEGIN("comm_small");

	cudaMalloc((void**)&d, n*sizeof(int));

	CALI_MARK_END("comm_small");

	CALI_MARK_BEGIN("comm_large_h2d");

	cudaMemcpy(d,a,n*sizeof(int),cudaMemcpyHostToDevice);

	CALI_MARK_END("comm_large_h2d");

	CALI_MARK_BEGIN("comp_large");

	for(i=0;i<n;i++){
		//int size=n/2;

		oddeven<<<n/2,1>>>(d,i%2,n);

	}
	printf("\n");

	CALI_MARK_END("comp_large");

	CALI_MARK_BEGIN("comm_large_d2h");

	cudaMemcpy(c,d,n*sizeof(int), cudaMemcpyDeviceToHost);

	CALI_MARK_END("comm_large_d2h");

	// check to see if the array is sorted
	CALI_MARK_BEGIN("correctness_check");

	int sorted = 1;
	for(i=0; i<n-1; i++)
	{
		if(c[i]>c[i+1])
		{
			sorted = 0;
			break;
		}
	}

	if(sorted)
		printf("Array is sorted\n");
	else
		printf("Array is not sorted\n");

	CALI_MARK_END("correctness_check");

	printf("Sorted Array is:\t");
	for(i=0; i<n; i++)
	{
		printf("%d\t",c[i]);
	}

	cudaFree(d);

	CALI_MARK_END("main");
	return 0;
}