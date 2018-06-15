/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This application demonstrates how to use the CUDA API to use multiple GPUs,
 * with an emphasis on simple illustration of the techniques (not on performance).
 *
 * Note that in order to detect multiple GPUs in your system you have to disable
 * SLI in the nvidia control panel. Otherwise only one GPU is visible to the
 * application. On the other side, you can still extend your desktop to screens
 * attached to both GPUs.
 */

// System includes
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include <timer.h>

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int MAX_GPU_COUNT = 2;
const float up_value    = 100.0;

typedef struct
{
    float *fai_h;
    float *send, *recv;
    float *fai_d, *fai_d_n;
    //send offset address, recieve offset address
    int SOA,ROA,HOA;      
    float *temp;
    cudaStream_t stream;
} TGPUG;


////////////////////////////////////////////////////////////////////////////////
// GPU calculation kernel
////////////////////////////////////////////////////////////////////////////////

__device__ float SingleFai(float *fai, unsigned int i,unsigned int j,size_t pitch) {
	float *a = (float*)((char*)fai + (i - 1)*pitch);
	float *b = (float*)((char*)fai + (i + 1)*pitch);
	float *c = (float*)((char*)fai + i*pitch);
	return ((a[j] + b[j] + c[j - 1] + c[j + 1]) / 4);
}


__global__ void SingleNodeFaiIter(float *fai,float *fai_n,size_t pitch,int H, int W, int flag) {
	//unsigned int i = blockDim.y*blockIdx.y + threadIdx.y;
	//unsigned int j = blockDim.x*blockIdx.x + threadIdx.x;
	for (int i = blockDim.y*blockIdx.y + threadIdx.y; i < H; i += blockDim.y*gridDim.y) {
		float *fai_row_n = (float*)((char*)fai_n + i*pitch);
		for (int j = blockDim.x*blockIdx.x + threadIdx.x; j < W; j += blockDim.x*gridDim.x) {
			if(flag==0){
				if (i > 1 && i < H - 1 && j > 0 && j < W - 1)
				fai_row_n[j] = SingleFai(fai, i, j, pitch);
			}
			else if(flag==1){
				if (i > 0 && i < H - 2 && j > 0 && j < W - 1)
				fai_row_n[j] = SingleFai(fai, i, j, pitch);
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// Data initialization
////////////////////////////////////////////////////////////////////////////////
void DataInitial(float *fai,unsigned int size, int H, int W)
{
	for(int i=0; i<size; i++){
		if(i>W-1 && i<2*W)
			fai[i]=up_value;
		else
			fai[i]=0;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Data saving
////////////////////////////////////////////////////////////////////////////////
int DataSave(float *fai, int M, int N) 
{
	char filename[100];
	strcpy(filename,"/public/home/wang_xiaoyue/data/fai_data.txt");
	ofstream f(filename);
	if (!f) {
		cout << "File open error!" << endl;
		return 0;
	}
	for (int i = 0; i < M*N; i++) {
		f << fai[i] << ' ';
		if ((i + 1) % N == 0)
			f << endl;
	}
	f.close();
	return 1;
}

////////////////////////////////////////////////////////////////////////////////
// Device information
////////////////////////////////////////////////////////////////////////////////
void GetDeviceName(count) 
{ 

    cudaDeviceProp prop;
    if (count== 0)
    {
        cout<< "There is no device."<< endl;
    }
    for(int i= 0;i< count;++i)
    {
        cudaGetDeviceProperties(&prop,i) ;
        cout << "Device "<<i<<" name is :" << prop.name<< endl;
    } 
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{

    TGPUG G[MAX_GPU_COUNT];
    int i,n;
    int GPU_N;
    size_t pitch;
    int H,W,DH,DW,DWP;
    float *fai_T;

    cout<<"\nPlease input grids number (height, width): "<<endl;
    cin>>H>>W;
   
    unsigned int HSIZE;

    const dim3 blockDim(32, 16,1);
	const dim3 gridDim(8, 8,1);


    cout<<"\nStarting MultiGPU"<<endl;
    checkCudaErrors(cudaGetDeviceCount(&GPU_N));

    if (GPU_N > MAX_GPU_COUNT){
        GPU_N = MAX_GPU_COUNT;
    }

    cout<<"CUDA-capable device count: "<<GPU_N<<endl;
    GetDeviceName(GPU_N);
  
	cout<<"Initializing data...\n"<<endl;    
    HSIZE = (H+2) * W;
    
    checkCudaErrors(cudaMallocHost((void**)&fai_T, HSIZE*sizeof(float)));
    DataInitial(fai_T, HSIZE, H, W);
    
    cout<<"Start timing..."<<endl;
    StartTimer();
    //Get data sizes for each GPU
    DH = H / GPU_N + 2;
    DW = W;
    //Subdividing total data across GPUs, Create streams for issuing GPU command asynchronously and allocate memory (GPU and System page-locked)
    
    for (i = 0; i < GPU_N; i++){

	    G[i].fai_h = fai_T + i * (H / GPU_N) * W;
	    //set device
        checkCudaErrors(cudaSetDevice(i));
        //creat stream
        checkCudaErrors(cudaStreamCreate(&G[i].stream));
        //Allocate memory
        checkCudaErrors(cudaMallocPitch((void **)&G[i].fai_d, &pitch, DW * sizeof(float), DH));
        checkCudaErrors(cudaMallocPitch((void **)&G[i].fai_d_n, &pitch, DW * sizeof(float), DH));

        checkCudaErrors(cudaMallocHost((void **)&G[i].send, DW * sizeof(float)));
        checkCudaErrors(cudaMallocHost((void **)&G[i].recv, DW * sizeof(float)));

        if(i==1){
        	G[i].SOA = pitch/sizeof(float);
        	G[i].ROA = 0;
        	G[i].HOA = (H/2 + 1)*W;
        }
        else{
        	G[i].SOA = (pitch/sizeof(float))*(DH-2);
        	G[i].ROA = (pitch/sizeof(float))*(DH-1);
        	G[i].HOA = W;
        }
    }

    DWP = pitch/sizeof(float);

    //Start compute on GPU(s)
    cout<<"Computing with "<<GPU_N<<" GPUs..."<<endl;    

    //Copy initial data to GPU
    for (i = 0; i < GPU_N; i++){
    	//Set device
        checkCudaErrors(cudaSetDevice(i));
        //Copy input data from CPU
		checkCudaErrors(cudaMemcpy2DAsync(G[i].fai_d, pitch, G[i].fai_h, DW * sizeof(float), DW * sizeof(float), DH, cudaMemcpyHostToDevice, G[i].stream));		
		checkCudaErrors(cudaMemcpy2DAsync(G[i].fai_d_n, pitch, G[i].fai_h, DW * sizeof(float), DW * sizeof(float), DH, cudaMemcpyHostToDevice, G[i].stream));
    }

    //Launch the kernel and copy boundary data back. All asynchronously
    for (n = 0; n < 5000; n++){
    	for (i = 0; i < GPU_N; i++){
    		//Set device
    		checkCudaErrors(cudaSetDevice(i));
    		//Perform GPU computations
        	SingleNodeFaiIter<<<gridDim, blockDim, 0, G[i].stream>>>(G[i].fai_d, G[i].fai_d_n, pitch, DH, DW, i);
        	//Read back boundary data from GPU
        	checkCudaErrors(cudaMemcpy2DAsync(G[i].send, DW * sizeof(float), G[i].fai_d_n + G[i].SOA, pitch, 
        									  DW * sizeof(float), 1, cudaMemcpyDeviceToHost, G[i].stream));
        }        
        	//getLastCudaError("SingleNodeFaiIter() execution failed.\n");
        

        for (i = 0; i < GPU_N; i++){
        	//Set device
        	checkCudaErrors(cudaSetDevice(i));
		    //Wait for all operations to finish
        	cudaStreamSynchronize(G[i].stream);
   
        	G[i].temp = G[i].fai_d;
        	G[i].fai_d = G[i].fai_d_n;
        	G[i].fai_d_n = G[i].temp;
        }
        	//Write new boundary value to GPU
        for (i = 0; i < GPU_N; i++){
        	//Set device
        	checkCudaErrors(cudaSetDevice(i));
        	int j=(i==0)?1:0;
			checkCudaErrors(cudaMemcpy2DAsync(G[i].fai_d + G[i].ROA, pitch, G[j].send, DW * sizeof(float), 
											  DW * sizeof(float), 1, cudaMemcpyHostToDevice, G[i].stream));        	
        }
        
    }


    for (i = 0; i < GPU_N; i++){
    	//Set device
    	checkCudaErrors(cudaSetDevice(i));
    	//Read back final data from GPU
    	checkCudaErrors(cudaMemcpy2DAsync(fai_T + G[i].HOA, W * sizeof(float), G[i].fai_d + DWP , pitch, 
										  DW * sizeof(float), DH-2, cudaMemcpyDeviceToHost, G[i].stream)); 
    }

    //Process GPU results
    for (i = 0; i < GPU_N; i++){
        //Set device
        checkCudaErrors(cudaSetDevice(i));
		//Wait for all operations to finish
        cudaStreamSynchronize(G[i].stream);

        //Shut down this GPU
        checkCudaErrors(cudaFreeHost(G[i].send));
        checkCudaErrors(cudaFreeHost(G[i].recv));
        checkCudaErrors(cudaFree(G[i].fai_d));
        checkCudaErrors(cudaFree(G[i].fai_d_n));
        checkCudaErrors(cudaStreamDestroy(G[i].stream));
    }

    cout<<"    GPU Processing time: "<<GetTimer()/1e3<<"(s)"<<endl;

    // Compute on Host CPU
    if(argc==2){
        cout<<"Saving data..."<<endl;
        if(DataSave(fai_T + W, H, W))
        cout<<"    Saving completed\n"<<endl;
    }

    //clean up
    checkCudaErrors(cudaFreeHost(fai_T));

    return 0;
}
