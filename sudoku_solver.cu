/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include<curand_kernel.h>
#include <math.h>
#include <cuda.h>
#include <cuda_profiler_api.h>

#define NUM_ITERATION 10000
#define INIT_TEMPERATURE 0.4
#define NUM_CHAINS 15
#define MIN_TEMPERATURE 0.001
#define INIT_TOLERANCE 1
#define DELTA_T 0.2

__constant__ int d_mask[81];
char outname[50];
//Error Checks

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


// Kernel for initializing random number generators
__global__ void init_random_generator(curandState *state) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;

   curand_init(1337, idx, 0, &state[idx]);
}

// This functions returns the count of number of unique elements in a row or column number according to the flag (Device Version)
__device__ int d_num_unique(int rc_num,int sudoku[][9],int flag)
{
	int nums[9]={1,2,3,4,5,6,7,8,9};
	int idx, unique_Count;

	    unique_Count = 0;

	    for(int j=0;j<9;j++)
	    {
	        if(flag==2)
	        	idx = sudoku[j][rc_num]-1;
	        else
	        	idx = sudoku[rc_num][j]-1;
	        if(idx==-1)
	            return -1;
	        if(nums[idx]!=0)
	        {
	            unique_Count+=1;
	            nums[idx]=0;
	        }
	    }

	    return unique_Count;
}

//Computes the energy by adding the number of unique elements in all the rows and columns
__device__ int d_compute_energy(int sudoku[][9])
{
	int energy=0;

	for(int i=0;i<9;i++)
	  energy += d_num_unique(i,sudoku,1) + d_num_unique(i,sudoku,2);

	return 162-energy;
}

//Kernel to run a Markov chain
__global__ void markov(int* sudoku,curandState *state,int cur_energy,float temperature,int *b1,int *b2,int *b3,int *b4,int *b5,int *b6,int *b7,int *b8,int *b9,int *b10,int *b11,int *b12,int *b13,int *b14,int *b15,int *energy_block)
{

	__shared__ int shd_sudoku[9][9];

	int thread_x=threadIdx.x;
	int thread_y=threadIdx.y;
	int thread_num_local= threadIdx.x*blockDim.x + threadIdx.y;
	int block_num= blockIdx.x*blockDim.x + blockIdx.y;

	//Bring the sudoku to shared memory

	shd_sudoku[thread_x][thread_y]=sudoku[thread_x+ 9*thread_y];


	if(thread_num_local!=0)
	{
		return;
	}

	int block_x;
	int block_y;
	int r1_x, r1_y, r2_x, r2_y;
	int temp;
	int energy;


	for(int iter=0;iter<NUM_ITERATION;iter++)
	{

		//Select a Random sub block in the sudoku
		block_x = 3*(int)(3.0*curand_uniform(&state[block_num]));
		block_y = 3*(int)(3.0*curand_uniform(&state[block_num]));

		//Select two unmasked points

		do
		{
			r1_x=(int)3.0*curand_uniform(&state[block_num]);
			r1_y=(int)3.0*curand_uniform(&state[block_num]);

		}while(d_mask[(block_x+r1_x)+9*(block_y+r1_y)]==1);


		do{
			r2_x=(int)3.0*curand_uniform(&state[block_num]);
			r2_y=(int)3.0*curand_uniform(&state[block_num]);

		}while(d_mask[(block_x+r2_x)+9*(block_y+r2_y)]==1);

		//Swap the elements

		temp=shd_sudoku[block_x+r1_x][block_y+r1_y];
		shd_sudoku[block_x+r1_x][block_y+r1_y]=shd_sudoku[block_x+r2_x][block_y+r2_y];
		shd_sudoku[block_x+r2_x][block_y+r2_y]=temp;

		//Compute the energy of this new state
		energy=d_compute_energy(shd_sudoku);


		if(energy<cur_energy)
			cur_energy = energy;

		else{

		//Accept the state

		if(exp((float)(cur_energy-energy)/temperature)>curand_uniform(&state[block_num]))
			cur_energy = energy;

//			if(cur_energy-energy>0.2)
//						cur_energy = energy;

		//Reject the state and undo changes
		else{
			temp=shd_sudoku[block_x+r1_x][block_y+r1_y];
			shd_sudoku[block_x+r1_x][block_y+r1_y]=shd_sudoku[block_x+r2_x][block_y+r2_y];
			shd_sudoku[block_x+r2_x][block_y+r2_y]=temp;
			}
		}

		//If reached the lowest point break
		if(energy==0)
			break;


	}

	//Write the result back to memory
	for(int i=0;i<9;i++)
	{
		for(int j=0;j<9;j++)
		{
			if(block_num==0)
				b1[i+9*j]=shd_sudoku[i][j];
			if(block_num==1)
				b2[i+9*j]=shd_sudoku[i][j];
			if(block_num==2)
				b3[i+9*j]=shd_sudoku[i][j];
			if(block_num==3)
				b4[i+9*j]=shd_sudoku[i][j];
			if(block_num==4)
				b5[i+9*j]=shd_sudoku[i][j];
			if(block_num==5)
				b6[i+9*j]=shd_sudoku[i][j];
			if(block_num==6)
				b7[i+9*j]=shd_sudoku[i][j];
			if(block_num==7)
				b8[i+9*j]=shd_sudoku[i][j];
			if(block_num==8)
				b9[i+9*j]=shd_sudoku[i][j];
			if(block_num==9)
				b10[i+9*j]=shd_sudoku[i][j];
			if(block_num==10)
				b11[i+9*j]=shd_sudoku[i][j];
			if(block_num==11)
				b12[i+9*j]=shd_sudoku[i][j];
			if(block_num==12)
				b13[i+9*j]=shd_sudoku[i][j];
			if(block_num==13)
				b14[i+9*j]=shd_sudoku[i][j];
			if(block_num==14)
				b15[i+9*j]=shd_sudoku[i][j];
		}
	}

	//Write the energy back to memory for the current state
	energy_block[block_num]=cur_energy;

}

//Display the sudoku
void display_sudoku(int *n){

    printf("\n_________________________\n");
    for(int i=0;i<9;i++){
        printf("| ");
        for(int j=0;j<9;j=j+3)
            printf("%1d %1d %1d | ",n[i+9*j],n[i+9*(j+1)],n[i+9*(j+2)]);
        if((i+1)%3==0){
            printf("\n-------------------------\n");
        }else printf("\n");
    }
    return;
}

/*Initialize the sudoku. 1) Read the partial sudoku.
					     2) Place values in all the empty slots such that the 3x3 subgrid clause is satisfied */
void init_sudoku(int *s,int *m,char* fname)
{
	FILE *fin ;
	fin = fopen(fname,"r");

	//Output file name
	int len;
	for(len=0;len<strlen(fname)-2;len++)
		outname[len]=fname[len];
	strcat(outname,"out");

	int in;

	int x, y;
	int p, q;
	int idx;

	int nums_1[9],nums_2[9];


	//Read the partial sudoku from file
	//Compute the mask. 0 -> mutable value 1-> non-mutable

	for(int i=0;i<9;i++){

		for(int j=0;j<9;j++){

			fscanf(fin,"%1d",&in);
			s[i+9*j] = in;
			if(in==0)
				m[i+9*j]=0;
			else
				m[i+9*j]=1;
			}
		}
	fclose(fin);

	printf("Puzzle\n");
	display_sudoku(s);

	//Place values in all the empty slots such that the 3x3 subgrid clause is satisfied
	for(int block_i=0;block_i<3;block_i++)
	{
		for(int block_j=0;block_j<3;block_j++)
		{
			for(int k=0;k<9;k++)
				nums_1[k]=k+1;

				for(int i=0;i<3;i++)
				{
					for(int j=0;j<3;j++)
					{
						x = block_i*3 + i;
						y = block_j*3 + j;

						if(s[x+9*y]!=0){
							p = s[x+9*y];
							nums_1[p-1]=0;
						}
					}
				}
				q = -1;
				for(int k=0;k<9;k++)
				{
					if(nums_1[k]!=0)
					{
						q+=1;
						nums_2[q] = nums_1[k];
					}
				}
				idx = 0;
				for(int i=0;i<3;i++)
				{
					for(int j=0;j<3;j++)
					{
						x = block_i*3 + i;
						y = block_j*3 + j;
						if(s[x+9*y]==0)
						{
							s[x+9*y] = nums_2[idx];
							idx+=1;
						}
					}
				}

			}
		}


//	int ar[3]={0,3,6};
//	int temp;
//
//
//	int rand1=random()%3;
//	int rand2=random()%3;
//
//	int r1_x,r1_y,r2_x,r2_y;
//	int block_x,block_y;

//for(int suf=0;suf<random()%20;suf++)
//{
//	block_x = ar[rand1];
//	block_y = ar[rand2];
//	do{
//		r1_x=random()%3;
//		r1_y=random()%3;;
//	}while(m[(block_x+r1_x)+9*(block_y+r1_y)]==1);
//
//
//	do{
//		r2_x=random()%3;;
//		r2_y=random()%3;;
//	}while(m[(block_x+r2_x)+9*(block_y+r2_y)]==1);
//
//	temp=s[(block_x+r1_x)+9*(block_y+r1_y)];
//	s[(block_x+r1_x)+9*(block_y+r1_y)]=s[(block_x+r2_x)+9*(block_y+r2_y)];
//	s[(block_x+r2_x)+9*(block_y+r2_y)]=temp;
//
//
//}


}

// This functions returns the count of number of unique elements in a row or column number according to the flag (Host Version)
int h_num_unique(int i, int k, int *n){

    int nums[9]={1,2,3,4,5,6,7,8,9};
    int idx, unique_count;

    unique_count = 0;

    for(int j=0;j<9;j++){

        if(k==1){
            idx = n[i+9*j]-1;
        }

        else{
            idx = n[j+9*i]-1;
        }

        if(idx==-1){
            return -1;
        }

        if(nums[idx]!=0){
            unique_count+=1;
            nums[idx]=0;
        }
    }
    return unique_count;
}

//Computes the energy by adding the number of unique elements in all the rows and columns
int h_compute_energy(int *n)
{
	    int energy = 0;

	    for(int i=0;i<9;i++){
	        energy += h_num_unique(i,1,n) + h_num_unique(i,2,n);
	    }

	    return 162 - energy;
}

void write_file(int *s)
{
	FILE *fout;
	fout=fopen(outname,"w");

	for(int i=0;i<9;i++)
	{
		for(int j=0;j<9;j++)
			fprintf(fout,"%1d",s[i+9*j]);
		if(i<8)
		fprintf(fout,"\n");
	}

	fclose(fout);

}

//Main
int main(int arg,char* argv[]) {


	//cudaSetDevice(1);
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	//Tunable Parameter
	int num_chains=NUM_CHAINS;
	float temperature=INIT_TEMPERATURE;
	float temp_min=MIN_TEMPERATURE;

	//Host pointers
	int *sudoku;
	int *mask;
	int *h_energy_host;

	int size=sizeof(int)*81;

	//Allocate memory
	gpuErrchk(cudaHostAlloc((void**)&sudoku,size,cudaHostAllocDefault));
	gpuErrchk(cudaHostAlloc((void**)&mask,size,cudaHostAllocDefault));
	gpuErrchk(cudaHostAlloc((void**)&h_energy_host,sizeof(int)*num_chains,cudaHostAllocDefault));
	init_sudoku(sudoku,mask,argv[1]);

	//Initial Energy of sudoku
	int current_energy=h_compute_energy(sudoku);

	printf("Current energy %d \n",current_energy);

	//Device pointers
	int *d_sudoku;
	int *d_b1,*d_b2,*d_b3,*d_b4,*d_b5,*d_b6,*d_b7,*d_b8,*d_b9,*d_b10,*d_b11,*d_b12,*d_b13,*d_b14,*d_b15;
	int *energy_block;
	//Allocate memory
	gpuErrchk(cudaMalloc((void**)&d_sudoku,size));
	gpuErrchk(cudaMalloc((void**)&d_mask,size));
	gpuErrchk(cudaMalloc((void**)&d_b1,size));
	gpuErrchk(cudaMalloc((void**)&d_b2,size));
	gpuErrchk(cudaMalloc((void**)&d_b3,size));
	gpuErrchk(cudaMalloc((void**)&d_b4,size));
	gpuErrchk(cudaMalloc((void**)&d_b5,size));
	gpuErrchk(cudaMalloc((void**)&d_b6,size));
	gpuErrchk(cudaMalloc((void**)&d_b7,size));
	gpuErrchk(cudaMalloc((void**)&d_b8,size));
	gpuErrchk(cudaMalloc((void**)&d_b9,size));
	gpuErrchk(cudaMalloc((void**)&d_b10,size));
	gpuErrchk(cudaMalloc((void**)&d_b11,size));
	gpuErrchk(cudaMalloc((void**)&d_b12,size));
	gpuErrchk(cudaMalloc((void**)&d_b13,size));
	gpuErrchk(cudaMalloc((void**)&d_b14,size));
	gpuErrchk(cudaMalloc((void**)&d_b15,size));
	gpuErrchk(cudaMalloc((void**)&energy_block,sizeof(int)*num_chains));

	//Copy Sudoku and Mask to GPU
	gpuErrchk(cudaMemcpy(d_sudoku,sudoku,size,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyToSymbol(d_mask,mask,size));

	//Grid and Block dimensions
	dim3 dimGrid(1,num_chains);
	dim3 dimBlock(9,9);

	printf("Solution");

	//Random number generators. Launch init_random_generator kernel
	curandState *d_state;
	gpuErrchk(cudaMalloc(&d_state, dimBlock.x* dimBlock.y * dimGrid.x * dimGrid.y));
	init_random_generator<<<dimGrid.x * dimGrid.y, dimBlock.x* dimBlock.y>>>(d_state);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());




	int tolerance=INIT_TOLERANCE;
	int min,min_idx;
	int e;

	int prev_energy=current_energy;

	cudaProfilerStart();



	//Simulated Annealing loop
	do{

		min=200;
		min_idx=200;
		markov<<< dimGrid,dimBlock >>>(d_sudoku,d_state,current_energy,temperature,d_b1,d_b2,d_b3,d_b4,d_b5,d_b6,d_b7,d_b8,d_b9,d_b10,d_b11,d_b12,d_b13,d_b14,d_b15,energy_block);
		gpuErrchk(cudaDeviceSynchronize());

		cudaMemcpy(h_energy_host,energy_block,sizeof(int)*num_chains,cudaMemcpyDeviceToHost);

		for(e=0;e<num_chains;e++)
		{
			if(h_energy_host[e]<min)
			{
				min=h_energy_host[e];
				min_idx=e;
			}

		}


		if(min_idx==0)
		{
			cudaMemcpy(d_sudoku,d_b1,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==1)
		{
			cudaMemcpy(d_sudoku,d_b2,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==2)
		{
			cudaMemcpy(d_sudoku,d_b3,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==3)
		{
			cudaMemcpy(d_sudoku,d_b4,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==4)
		{
			cudaMemcpy(d_sudoku,d_b5,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==5)
		{
			cudaMemcpy(d_sudoku,d_b6,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==6)
		{
			cudaMemcpy(d_sudoku,d_b7,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==7)
		{
			cudaMemcpy(d_sudoku,d_b8,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==8)
		{
			cudaMemcpy(d_sudoku,d_b9,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==9)
		{
			cudaMemcpy(d_sudoku,d_b10,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==10)
		{
			cudaMemcpy(d_sudoku,d_b11,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==11)
		{
			cudaMemcpy(d_sudoku,d_b12,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==12)
		{
			cudaMemcpy(d_sudoku,d_b13,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==13)
		{
			cudaMemcpy(d_sudoku,d_b14,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==14)
		{
			cudaMemcpy(d_sudoku,d_b15,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}


		if(current_energy==0)
		{

			break;
		}

		if(current_energy==prev_energy)
			tolerance--;
		else
			tolerance=INIT_TOLERANCE;

		// Random restart if energy is stuck
		if(tolerance<0)
		{
			//printf("Randomizing\n");
			cudaMemcpy(sudoku,d_sudoku,size,cudaMemcpyDeviceToHost);

			int ar[3]={0,3,6};
				int tempa;
				int rand1=random()%3;
				int rand2=random()%3;

				int r1_x,r1_y,r2_x,r2_y;
				int block_x,block_y;

			for(int suf=0;suf<random()%10;suf++)
			{
				block_x = ar[rand1];
				block_y = ar[rand2];
				do{
					r1_x=random()%3;
					r1_y=random()%3;;
				}while(mask[(block_x+r1_x)+9*(block_y+r1_y)]==1);

				do{
					r2_x=random()%3;;
					r2_y=random()%3;;
				}while(mask[(block_x+r2_x)+9*(block_y+r2_y)]==1);

				tempa=sudoku[(block_x+r1_x)+9*(block_y+r1_y)];
				sudoku[(block_x+r1_x)+9*(block_y+r1_y)]=sudoku[(block_x+r2_x)+9*(block_y+r2_y)];
				sudoku[(block_x+r2_x)+9*(block_y+r2_y)]=tempa;
			}
			cudaMemcpy(d_sudoku,sudoku,size,cudaMemcpyHostToDevice);
			current_energy=h_compute_energy(sudoku);
			//printf("Energy after randomizing %d \n",current_energy);
			tolerance=INIT_TOLERANCE;
			temperature=temperature+DELTA_T;
		}

		prev_energy=current_energy;

		if(current_energy==0)
		{
			break;
		}
		temperature=temperature*0.8;

		//printf("Energy after temp %f is %d \n",temperature,current_energy);


	}while(temperature>temp_min);

	cudaProfilerStop();

	cudaMemcpy(sudoku,d_sudoku,size,cudaMemcpyDeviceToHost);

	display_sudoku(sudoku);

	write_file(sudoku);

	current_energy=h_compute_energy(sudoku);

	printf("Current energy %d \n",current_energy);

	return 0;
}




