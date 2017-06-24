#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define POLYGON_SIZE 3
#define BOOL int
#define FALSE 0
#define TRUE 1
#define UINT unsigned int

struct Point2F
{
	float x, y;
};

struct Polygon
{
	Point2F points[POLYGON_SIZE];
};

struct PolygonCount
{
	unsigned int count_ccw, count_cw;
};

cudaError_t countPolygonsWithCuda(
	const Polygon* polygons,
	const UINT polygonCount,
	const UINT num_threads,
	UINT& count_ccw,
	UINT& count_cw);
void countPolygonsSequencial(
	const Polygon* polygons,
	const UINT polygonCount,
	UINT& count_ccw,
	UINT& count_cw);
float randSingle(float max);
void genRndPolygons(Polygon* first, const UINT length);
void runPerformanceTest(const UINT polygonCount, const UINT gpuThreads);

__host__  __device__ BOOL isPolygonCCW(const Polygon& polygon)
{
	// See http://www.geeksforgeeks.org/orientation-3-ordered-points/
	// for details of below formula.
	float val = (polygon.points[1].x - polygon.points[0].y) * (polygon.points[2].x - polygon.points[1].x) -
		(polygon.points[1].x - polygon.points[0].x) * (polygon.points[2].y - polygon.points[1].y);

	if (val == 0)
		return TRUE;  // colinear

	return (val > 0) ? FALSE : TRUE; // clock or counterclock wise
}

__global__ void countPolygonsKernel(const Polygon* polygons, PolygonCount* res, const unsigned int div_size)
{
	int start = threadIdx.x * div_size;

	unsigned int count_ccw = 0;
	unsigned int count_cw = 0;

	for (int i = 0; i < div_size; i++)
	{
		int cur = start + i;

		// Check if polygon is ccw or cw.
		if (isPolygonCCW(polygons[cur]) == TRUE)
			count_ccw++;
		else
			count_cw++;
	}

	res[threadIdx.x].count_ccw = count_ccw;
	res[threadIdx.x].count_cw = count_cw;
}


int main()
{
	runPerformanceTest(10, 10);

	runPerformanceTest(100, 100);
	runPerformanceTest(1000, 1000);
	runPerformanceTest(10000, 1000);
	runPerformanceTest(100000, 1000);
	runPerformanceTest(1000000, 1000);
	runPerformanceTest(10000000, 1000);

	getchar();

    return 0;
}

float randSingle(float max)
{
	return ((float)rand() / (float)RAND_MAX) * max;
}

void genRndPolygons(Polygon* first, const UINT length)
{
	for (UINT i = 0; i < length; i++)
	{
		for (UINT y = 0; y < POLYGON_SIZE; y++)
		{
			first[i].points[y].x = randSingle(100.0);
			first[i].points[y].y = randSingle(100.0);
		}
	}
}


void runPerformanceTest(const UINT polygonCount, const UINT gpuThreads)
{
	// Generate polygons
	Polygon* polygons = (Polygon*)malloc(sizeof(Polygon) * polygonCount);
	genRndPolygons(polygons, polygonCount);

	unsigned int count_ccw, count_cw;

	// Parallel
	clock_t t_begin = clock();
	cudaError_t cudaStatus = countPolygonsWithCuda(polygons, polygonCount, gpuThreads, count_ccw, count_cw);
	clock_t t_end = clock();
	double time_spent_gpu = (double)(t_end - t_begin) / CLOCKS_PER_SEC;

	// Sequencial 
	t_begin = clock();
	countPolygonsSequencial(polygons, polygonCount, count_ccw, count_cw);
	t_end = clock();
	double time_spent_sequential = (double)(t_end - t_begin) / CLOCKS_PER_SEC;

	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "addWithCuda failed!");
	else
	{
		printf("Performance with polygon count %d\n", polygonCount);
		//printf("CCW: %d;   CW: %d\n", count_ccw, count_cw);
		printf("\tGPU time: %f\n", time_spent_gpu);
		printf("\tSequential time: %f\n\n", time_spent_sequential);
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceReset failed!");

	free(polygons);
}

cudaError_t countPolygonsWithCuda(
	const Polygon* polygons,
	const UINT polygonCount,
	const UINT num_threads,
	UINT& count_ccw,
	UINT& count_cw)
{
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(0);

	if (polygonCount % num_threads)
		fprintf(stderr, "cudaSetDevice failed!  Polygon count must be dividable by number of threads.");
	else if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	else
	{
		Polygon* dev_poly;
		unsigned int dev_poly_mem_size = sizeof(Polygon) * polygonCount;

		cudaStatus = cudaMalloc<Polygon>(&dev_poly, dev_poly_mem_size);
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "cudaMalloc failed!");
		else
		{
			cudaStatus = cudaMemcpy(dev_poly, polygons, dev_poly_mem_size, cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "cudaMemcpy failed!");
			else
			{
				PolygonCount* dev_res;
				size_t dev_res_mem_size = sizeof(PolygonCount) * num_threads;

				cudaStatus = cudaMalloc<PolygonCount>(&dev_res, dev_res_mem_size);
				if (cudaStatus != cudaSuccess)
					fprintf(stderr, "cudaMalloc failed!");
				else
				{
					unsigned int div_size = polygonCount / num_threads;

					countPolygonsKernel <<<1, num_threads>>>(dev_poly, dev_res, div_size);

					// Check for any errors launching the kernel
					cudaStatus = cudaGetLastError();
					if (cudaStatus != cudaSuccess)
						fprintf(stderr, "countPolygons launch failed: %s\n", cudaGetErrorString(cudaStatus));
					else
					{
						PolygonCount* res = (PolygonCount*)malloc(sizeof(PolygonCount) * num_threads);
						cudaStatus = cudaMemcpy(res, dev_res, dev_res_mem_size, cudaMemcpyDeviceToHost);
						if (cudaStatus != cudaSuccess)
							fprintf(stderr, "cudaMemcpy failed! %s", cudaGetErrorString(cudaStatus));
						else
						{
							count_ccw = 0;
							count_cw = 0;

							for (UINT i = 0; i < num_threads; i++)
							{
								count_ccw += res[i].count_ccw;
								count_cw += res[i].count_cw;
							}
						}

						free(res);
					}

					// Free memory after execution
					cudaFree(dev_res);
				}
			}

			// Free memory after execution
			cudaFree(dev_poly);
		}
	}

	return cudaStatus;
}

void countPolygonsSequencial(
	const Polygon* polygons,
	const UINT polygonCount,
	UINT& count_ccw,
	UINT& count_cw)
{
	count_ccw = 0;
	count_cw = 0;

	for (UINT i = 0; i < polygonCount; i++)
	{
		if (isPolygonCCW(polygons[i]) == TRUE)
			count_ccw++;
		else
			count_cw++;
	}
}
