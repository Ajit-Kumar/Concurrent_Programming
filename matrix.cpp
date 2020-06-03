#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<time.h>

#define MAX_DIM 2000*2000

typedef double TYPE;
TYPE ONEDA[MAX_DIM];
TYPE ONEDB[MAX_DIM];

/*
functions
*/
void generate_random_matrix(int r,int c,int **a,int **b);
void convert2d_to_1d(int r,int c,int **a,int **b);
void matrix_multiplication(int r,int c,int **a ,int **b,int **mul);
void parallel_matrix_multiplication(int r,int c,int **a ,int **b,int **mul);
void optimized_matrix_multiplication(int r,int c,int **a ,int **b,int **mul);
void resultant_matrix(int r,int c,int **mul);

int main()
{	
    int r,c;

	printf("Matrix Multiplication\n");
	
	printf("Enter the dimensions of  matrix = ");
	scanf("%d",&r);
	
	if(r>0)
	{
    	c = r;
	
	/*
	dynamic memory allocation
	*/
		int **a = (int **)malloc(r * sizeof(int *)); 
		int **b =  (int **)malloc(r * sizeof(int *));
		int **mul =  (int **)malloc(r * sizeof(int *));
	
		for(int i=0;i<r;i++)
		{
			a[i] = (int *)malloc(r * sizeof(int)); 
			b[i] = (int *)malloc(r * sizeof(int)); 
			mul[i] = (int *)malloc(r * sizeof(int));
		}
	
	/*
	generation of random matrix
	*/
		generate_random_matrix(r,c,a,b);
	
    /*
    Normal Matrix multiplication
    */
    
    	matrix_multiplication(r,c,a,b,mul);
    
    /*
    Parallel Matrix Multiplication
	*/
	
		parallel_matrix_multiplication(r,c,a,b,mul);
	
	/*
	convert 2D matrix to 1D matrix
	*/
	
		convert2d_to_1d(r,c,a,b);
    
    /*
    optimized matrix multiplication
	*/
	
		optimized_matrix_multiplication(r,c,a,b,mul);
	
	/*
	resultant matrix
	*/
	
	//resultant_matrix(r,c,mul);
		
	/*
	dynamically de-allocating the memory.
	*/
		free(a);
		free(b);
		free(mul);
	}
else
{
	printf("Enter correct dimension of matrix\n");
}
	return 0;
}

void generate_random_matrix(int r,int c,int **a, int **b)
{
	#pragma omp parallel for
	for(int i=0;i<r;i++)
	{
		for(int j=0;j<c;j++)
		{
			a[i][j] = rand()%100;
    		b[i][j] = rand()%100;
		}
	//printf("\n");
	}
	
	/*
	printing first random matrix
	
    printf("The First Matrix A = \n");
    for(int i=0;i<r;i++)
	{
		for(int j=0;j<c;j++)
		{
			printf("%d\t",a[i][j]);    
		}
		printf("\n");
	}
	*/
	
	/*
	printing of second random matrix
	
	printf("The First Matrix B = \n");
    for(int i=0;i<r;i++)
	{
		for(int j=0;j<c;j++)
		{
			printf("%d\t",b[i][j]);    
		}
		printf("\n");
	}
	*/
}

/*
  Sequential Matrix Multiplication Algorithm using IJK Algorithm
*/
void matrix_multiplication(int r,int c,int **a ,int **b,int **mul)
{

	clock_t begin = clock();
	
	for(int i=0;i<r;i++)
		{
			for(int j=0;j<c;j++)
			{
				mul[i][j]=0;  
				for(int k=0;k<c;k++)
				{
					mul[i][j] +=  a[i][k] * b[k][j];
				}
			}
		}
	clock_t end = clock();
	
	double time_taken = (double)(end - begin) / CLOCKS_PER_SEC;
	
	printf("The Normal matrix multiplication took %f seconds to execute\n", time_taken);
}

/*
  Parallelised Matrix Multiplication Algorithm using IJK Algorithm
*/
void parallel_matrix_multiplication(int r,int c,int **a ,int **b,int **mul)//ijk algorithm
{

	clock_t begin = clock();
	#pragma omp parallel for
	for(int i=0;i<r;i++)
		{
			for(int j=0;j<c;j++)
			{ 
			    mul[i][j]=0;  
				for(int k=0;k<c;k++)
				{
					mul[i][j] += a[i][k] * b[k][j];
				}
			}
		}
	clock_t end = clock();
	
	double time_taken = (double)(end - begin) / CLOCKS_PER_SEC;
	
	printf("The Parallel matrix multiplication took %f seconds to execute\n", time_taken);
}

void convert2d_to_1d(int r,int c,int **a,int **b)
{   
	#pragma omp parallel for
	for(int i=0; i<r; i++)
	{
		for(int j=0; j<c; j++)
		{
			ONEDA[i * r + j] = a[i][j];
			ONEDB[j * c + i] = b[i][j];
		}
	}
}

/*
  Optimized Parallelised Matrix Multiplication Algorithm using IKJ Algorithm
*/
void optimized_matrix_multiplication(int r,int c,int **a ,int **b,int **mul)//ikj algorithm
{

	int i,j,newi,newj,k,total;
    
	clock_t begin = clock();
	convert2d_to_1d(r,c,a,b);
	#pragma omp parallel shared(mul) private(i,j,k,newi,newj,total) num_threads(40)
	{
		#pragma omp for schedule(static)
		for(i=0;i<r;i++)
		{
			newi = i * r;
			for(j=0;j<c;j++)
			{ 
				newj = j * c;
				total=0;  
				for(k=0;k<c;k++)
				{
					//total += a[i][k] * b[k][j];
					  total += ONEDA[newi + k] * ONEDB[newj + k];
				}
			mul[i][j] = total;
			}
		}
	}
	
	clock_t end = clock();
	
	double time_taken = (double)(end - begin) / CLOCKS_PER_SEC;
	
	printf("The Optimized parallel matrix multiplication took %f seconds to execute\n", time_taken);
}

/*
void resultant_matrix(int r,int c,int **mul)
{
	printf("Resultant Matrix of %d X %d is \n",r,c);
	for(int i=0;i<r;i++)
	{
		for(int j=0;j<c;j++)
		{
			printf("%d\t",mul[i][j]);    
		}
		printf("\n");
	}	
}
*/
