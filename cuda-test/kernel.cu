#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define POLYGON_SIZE 3
#define BOOL int
#define FALSE 0
#define TRUE 1

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
	unsigned int polygonCount,
	unsigned int num_threads,
	unsigned int& count_ccw,
	unsigned int& count_cw);
int randInt(int max);
float randSingle(float max);
void genRndPolygons(Polygon* first, const int length);
__host__ __device__ BOOL isPolygonCCW(const Polygon& polygon);

__host__ __device__ BOOL isPolygonCCW(const Polygon& polygon)
{
	// See http://www.geeksforgeeks.org/orientation-3-ordered-points/
	// for details of below formula.
	float val = (polygon.points[1].x - polygon.points[0].y) * (polygon.points[2].x - polygon.points[1].x) -
		(polygon.points[1].x - polygon.points[0].x) * (polygon.points[2].y - polygon.points[1].y);

	if (val == 0)
		return TRUE;  // colinear

	return (val > 0) ? FALSE : TRUE; // clock or counterclock wise
}

__global__ void countPolygons(const Polygon* polygons, PolygonCount* res, unsigned int div_size)
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
	const int polygonCount = 1000;
	Polygon polygons[polygonCount];

	genRndPolygons(polygons, polygonCount);
	
	unsigned int count_ccw, count_cw;

	clock_t t_begin = clock();

	// Add vectors in parallel.
	cudaError_t cudaStatus = countPolygonsWithCuda(polygons, polygonCount, 10, count_ccw, count_cw);

	clock_t t_end = clock();
	double time_spent = (double)(t_end - t_begin) / CLOCKS_PER_SEC;

	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	else
	{
		printf("CCW: %d;   CW: %d\n", count_ccw, count_cw);
		printf("Time: %f", time_spent);
	}

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
	{
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	getchar();

    return 0;
}

int randInt(int max)
{
	return rand() % max;
}

float randSingle(float max)
{
	return ((float)rand() / (float)RAND_MAX) * max;
}

void genRndPolygons(Polygon* first, const int length)
{
	for (int i = 0; i < length; i++)
	{
		for (int y = 0; y < POLYGON_SIZE; y++)
		{
			first[i].points[y].x = randSingle(100.0);
			first[i].points[y].y = randSingle(100.0);
		}
	}
}

cudaError_t countPolygonsWithCuda(
	const Polygon* polygons,
	const unsigned int polygonCount,
	const unsigned int num_threads,
	unsigned int& count_ccw,
	unsigned int& count_cw)
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

					countPolygons<<<1, num_threads>>>(dev_poly, dev_res, div_size);

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

							for (int i = 0; i < num_threads; i++)
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


