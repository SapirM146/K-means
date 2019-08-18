#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include "MainHeader.h"
#include "CudaHeader.h"


#define MASTER 0 
#define MAX_CUDA_BLOCKS_IN_GRID 100
#define CUDA_THREADS_IN_BLOCK 1000
#define NOT_CHANGED 0
#define CHANGED 1
#define NO_CLUSTERS_FOUND 0
#define CLUSTERS_FOUND 1
#define MASTER_START_TIME 0
#define NOT_BEST_FINISH_TIME 0
#define BEST_FINISH_TIME 1
#define POINT_FIELDS 5
#define CPOINT_FIELDS 2
#define KEEP_WORKING 1
#define STOP_WORKING 0

#define FILE_NAME_READ "Input\\input.txt" //***** please enter the full file path
#define FILE_NAME_WRITE "Output\\output.txt" //***** please enter the full file path



int main(int argc, char *argv[])
{
	int myid, numprocs;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	MPI_Status status;
	Point* arrPoint = NULL;
	int n, k, limit;
	double t, dt, qM, finish_time, finish_quality;
	int best_Result_Rank = MASTER, result = NO_CLUSTERS_FOUND;
	clock_t t1, t2;
	cPoint* arrCenterCluster;

	//Creating a new MPI data type for Point
	Point point;
	MPI_Datatype MPI_POINT;
	MPI_Datatype typeP[POINT_FIELDS] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT };
	int blocklenP[5] = { 1, 1, 1, 1, 1 };
	MPI_Aint dispP[5];
	
	// Create MPI user data type for Point
	dispP[0] = (char*)&point.x - (char*)&point;
	dispP[1] = (char*)&point.y - (char*)&point;
	dispP[2] = (char*)&point.vx - (char*)&point;
	dispP[3] = (char*)&point.vy - (char*)&point;
	dispP[4] = (char*)&point.cPoint_index - (char*)&point;
	MPI_Type_create_struct(POINT_FIELDS, blocklenP, dispP, typeP, &MPI_POINT);
	MPI_Type_commit(&MPI_POINT);

	//Creating a new MPI data type for cPoint
	cPoint cPoint;
	MPI_Datatype MPI_CPOINT;
	MPI_Datatype typeCP[CPOINT_FIELDS] = { MPI_DOUBLE, MPI_DOUBLE };
	int blocklenCP[2] = { 1, 1 };
	MPI_Aint dispCP[2];

	// Create MPI user data type for cPoint
	dispCP[0] = (char*)&cPoint.x - (char*)&cPoint;
	dispCP[1] = (char*)&cPoint.y - (char*)&cPoint;

	MPI_Type_create_struct(CPOINT_FIELDS, blocklenCP, dispCP, typeCP, &MPI_CPOINT);
	MPI_Type_commit(&MPI_CPOINT);

	if (numprocs < 1) 
	{
		printf("number of proccess need to be at least 1");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if (myid == MASTER) 
	{
		int i;
		printf("size of int = %zd , size of double = %zd , size of Point = %zd , size of cPoint = %zd\n\n", sizeof(int), sizeof(double), sizeof(Point), sizeof(cPoint));
		t1 = clock();
		read_From_File(FILE_NAME_READ, &arrPoint, &n, &k, &t, &dt, &limit, &qM);

		printf("data:\n");
		printf("n = %d, k = %d, t= %.4g, dt= %.4g, limit= %d, qM= %.4g\n\n", n, k, t, dt, limit, qM);
		printf("Calculating...\n\n");
		fflush(stdout);
		
		// Send the parameters for kmeans to slaves
		for (i = 1; i < numprocs; i++) {
			MPI_Send(&n, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(arrPoint, n, MPI_POINT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&k, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&t, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			MPI_Send(&dt, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			MPI_Send(&limit, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&qM, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		}
	}
	else // Slaves
	{
		// Recive the parameters for kmeans from master
		MPI_Recv(&n, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
		arrPoint = (Point*)malloc(sizeof(Point)*(n));
		MPI_Recv(arrPoint, n, MPI_POINT, MASTER, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&k, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&t, 1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&dt, 1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&limit, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&qM, 1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, &status);

		change_Points_Positions(arrPoint, n, dt*myid);
		copy_Points_To_Source(arrPoint, n);
	}

	const double initial_time = dt*myid;
	const double mpi_dt = dt*numprocs;

	arrCenterCluster = k_means(arrPoint, n, k, t, mpi_dt, limit, qM, initial_time, &finish_time, &finish_quality, &result, &best_Result_Rank, myid, numprocs);

	if (myid != MASTER) // Slaves
	{
		MPI_Recv(&best_Result_Rank, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
		if (best_Result_Rank == myid) 
		{
			MPI_Send(&finish_time, 1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
			MPI_Send(&finish_quality, 1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
			MPI_Send(arrCenterCluster, k, MPI_CPOINT, MASTER, 0, MPI_COMM_WORLD);
		}
	}
	else // MASTER
	{
		int i;
		for (i = 1; i < numprocs; i++)
			MPI_Send(&best_Result_Rank, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
		if (best_Result_Rank != MASTER) {
			MPI_Recv(&finish_time, 1, MPI_DOUBLE, best_Result_Rank, 0, MPI_COMM_WORLD, &status);
			MPI_Recv(&finish_quality, 1, MPI_DOUBLE, best_Result_Rank, 0, MPI_COMM_WORLD, &status);
			MPI_Recv(arrCenterCluster, k, MPI_CPOINT, best_Result_Rank, 0, MPI_COMM_WORLD, &status);
		}
		
		if(result == CLUSTERS_FOUND)
			write_To_Output_File(FILE_NAME_WRITE, arrCenterCluster, k, finish_time, finish_quality); // write to output file
		
		t2 = clock();
		float diff = ((float)(t2 - t1) / 1000000.0F) * 1000;

		if (result == NO_CLUSTERS_FOUND)
			printf("\nDidn't found clusters. Done after %f seconds.\n", diff);
		else
			printf("\nFound clusters. Done after %f seconds\n", diff);
		fflush(stdout);
	}

	free(arrPoint);
	free(arrCenterCluster);
	MPI_Finalize();
	return 0;
}

void read_From_File(const char* fileName, Point** arrPoint, int* n, int* k, double* t, double* dt, int* limit, double* qM)
{
	int i;
	FILE *f;
	errno_t err;
	err = fopen_s(&f, fileName, "r");
	if (err != 0)
	{
		printf("file corrupt or try to enter absolute path to txt file in global variable: filePath");
		exit(1);
	}

	// read parameters from file
	fscanf_s(f, "%d %d %lf %lf %d %lf\n", n, k, t, dt, limit, qM);

	*arrPoint = (Point*)malloc(sizeof(Point)*(*n));

	// read points from file
	for (i = 0; i < *n; i++)
	{
		fscanf_s(f, "%lf %lf %lf %lf\n", &(*arrPoint)[i].x, &(*arrPoint)[i].y, &(*arrPoint)[i].vx, &(*arrPoint)[i].vy);
	}

	fclose(f);

	if (k <= 0) {
		printf("\nNumber of clusters is 0\n");
		exit(1);
	}
}

/*
n - number of points
k - number of clusters to find
t – defines the end of time interval[0, T]
dt – defines moments t = n*dT, n = { 0, 1, 2, … , T / dT } for which calculate the clusters and the quality
limit – the maximum number of iterations for K - MEAN algorithm.
qM – quality measure to stop
*/

cPoint* k_means(Point* arrPoint, const int n, const int k, const double t, const double dt, const int limit, const double qM, const double initial_time, double* finish_time, double* finish_quality, int* result, int* best_Result_Rank, const int myid, const int numprocs)
{
	int i;
	int flag; // 0 = not point changed to other cPoint | 1 = at least one point changed to othe cPoint
	int workstatus, other_result;
	cPoint* arrCenterCluster = (cPoint*)malloc(k * sizeof(cPoint));
	double current_time = initial_time;
	MPI_Status status;

	do // current time is [0,t]
	{
		//time prints
		printf("myid = %d , current time = %lf\n", myid, current_time);
		fflush(stdout);

		// start CUDA to calculate the points positions changes of the next dt
		if (current_time < t)
			change_Points_Positions(arrPoint, n, dt);

		// initial cluster points
		#pragma omp parallel for 
		for (i = 0; i < k; i++)
		{
			arrCenterCluster[i].x = arrPoint[i].x;
			arrCenterCluster[i].y = arrPoint[i].y;
		}

		// loop until limit
		for (i = 0; i < limit; i++)
		{
			flag = NOT_CHANGED;

			//associate each Point to cPoint
			Cluster_Point_Min_Distance(arrPoint, n, arrCenterCluster, k, &flag);

			//calculate new avg of each cluster 
			calculateNewAvg(arrPoint, n, arrCenterCluster, k);
			if (flag == NOT_CHANGED)
				break;
		}

		// calculate quality
		double quality = calcQuality(arrPoint, n, arrCenterCluster, k);

		// check quality
		if (quality < qM) {
			*result = CLUSTERS_FOUND;
			*finish_time = current_time;
			*finish_quality = quality;
		}

		// mpi stop or keep working
		
		if (myid != MASTER) // slaves
		{
			MPI_Send(result, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
			MPI_Recv(&workstatus, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
			if (workstatus == STOP_WORKING)
				break;
		}
		else // master
		{
			for (i = 1; i < numprocs; i++)
			{
				if (current_time + i <= t)
				{
					MPI_Recv(&other_result, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);

					if (*result == CLUSTERS_FOUND) // master found clusters -> stop slaves
					{
						workstatus = STOP_WORKING;
						MPI_Send(&workstatus, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
					}
					else // master didn't found clusters
					{
						if (other_result == CLUSTERS_FOUND)
						{
							*best_Result_Rank = i;
							*result = other_result;
							workstatus = STOP_WORKING;
							MPI_Send(&workstatus, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
						}
						else
						{
							workstatus = KEEP_WORKING;
							MPI_Send(&workstatus, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
						}
					}
				}
			}
			if (*result == CLUSTERS_FOUND) // master found clusters -> stop master loop
				break;
		}
		
		// next dt
		current_time += dt;
		if (current_time < t)
			copy_Points_To_Source(arrPoint, n);

	} while (current_time <= t);
	
	return arrCenterCluster;
}

// associate a point with the closest cPoint
void Cluster_Point_Min_Distance(Point* points, const int points_size, cPoint* arrCenterCluster, const int arrCluster_size, int* flag)
{
	int i;
	#pragma omp parallel for
	for (i = 0; i < points_size; i++)
	{
		float minDistance = calcPointDistance(points[i].x, points[i].y, arrCenterCluster[0].x, arrCenterCluster[0].y);
		int j, minCPoint = 0;
		
		for (j = 1; j < arrCluster_size; j++)
		{
			float tmp = calcPointDistance(points[i].x, points[i].y, arrCenterCluster[j].x, arrCenterCluster[j].y);

			if (tmp < minDistance)
			{
				minDistance = tmp;
				minCPoint = j;
			}
		}
		if (points[i].cPoint_index != minCPoint) {
			*flag = CHANGED;
			points[i].cPoint_index = minCPoint;
		}
	}
}

// calculate Distance between 2 points (or cPoints) and returns the result
float calcPointDistance(const double x1, const double y1, const double x2, const double y2)
{
	double x = x2 - x1;
	double y = y2 - y1;
	return (float)sqrt((x*x) + (y*y));
}

// calculate a new avg for each cPoint
void calculateNewAvg(Point* arrPoint, const int sizeArrPoint, cPoint* arrCenterCluster, const int sizeArrCPoint)
{
	int i, numThreads;
	int* arr_numOfPoints_Threads;
	double* arr_sumX_Threads;
	double* arr_sumY_Threads;
	int* arr_numOfPoints = (int*)calloc(sizeArrCPoint, sizeof(int));
	double* arr_sumX = (double*)calloc(sizeArrCPoint, sizeof(double));
	double* arr_sumY = (double*)calloc(sizeArrCPoint, sizeof(double));

	#pragma omp parallel
	{
		int j, tid = omp_get_thread_num();
		double avgX, avgY;

		#pragma omp single
		{
			numThreads = omp_get_num_threads();
			arr_numOfPoints_Threads = (int*)calloc(numThreads*sizeArrCPoint, sizeof(int));
			arr_sumX_Threads = (double*)calloc(numThreads*sizeArrCPoint, sizeof(double));
			arr_sumY_Threads = (double*)calloc(numThreads*sizeArrCPoint, sizeof(double));
		}

		// suming Points and counting for avg
		#pragma omp for
		for (i = 0; i < sizeArrPoint; i++)
		{
			int tmp = arrPoint[i].cPoint_index;
			arr_numOfPoints_Threads[(tid*sizeArrCPoint) + tmp]++;
			arr_sumX_Threads[(tid*sizeArrCPoint) + tmp] += arrPoint[i].x;
			arr_sumY_Threads[(tid*sizeArrCPoint) + tmp] += arrPoint[i].y;
		}

		#pragma omp for
		for (i = 0; i < sizeArrCPoint; i++) // summing from threads arrs
		{
			for (j = 0; j < numThreads*sizeArrCPoint; j += sizeArrCPoint)
			{
				arr_numOfPoints[i] += arr_numOfPoints_Threads[i + j];
				arr_sumX[i] += arr_sumX_Threads[i + j];
				arr_sumY[i] += arr_sumY_Threads[i + j];
			}
		}
	
		#pragma omp for
		for (i = 0; i < sizeArrCPoint; i++)
		{
			if (arr_numOfPoints[i] != 0) // if cPoint has no points -> than keep it
			{
				avgX = arr_sumX[i] / arr_numOfPoints[i];
				avgY = arr_sumY[i] / arr_numOfPoints[i];

				// new Center?
				if (arrCenterCluster[i].x != avgX || arrCenterCluster[i].y != avgY)
				{
					arrCenterCluster[i].x = avgX;
					arrCenterCluster[i].y = avgY;
				}
			}
		}
	}

	free(arr_numOfPoints_Threads);
	free(arr_sumX_Threads);
	free(arr_sumY_Threads);
	free(arr_numOfPoints);
	free(arr_sumX);
	free(arr_sumY);
}

// calculate the quality and returns the result 
double calcQuality(const Point* points, const int size_Points, cPoint* cPoints, const int size_cPoints)
{
	double quality, counter, sum = 0;
	int i, numThreads;
	double* arr_Sum_Threads;
	double* arr_max_diameter;

	arr_max_diameter = calc_cPoint_Diameter(points, size_Points, cPoints, size_cPoints);

	#pragma omp parallel
	{
		int j, tid = omp_get_thread_num();
		double distance;

		#pragma omp single
		{
			numThreads = omp_get_num_threads();
			arr_Sum_Threads = (double*)calloc(numThreads, sizeof(double));
		}

		#pragma omp for
		for (i = 0; i < size_cPoints; i++)
		{
			for (j = 0; j < size_cPoints; j++)
			{
				if (j != i) // cPoint i is diffrent from cPoint j
				{
					distance = calcPointDistance(cPoints[i].x, cPoints[i].y, cPoints[j].x, cPoints[j].y);
					arr_Sum_Threads[tid] += arr_max_diameter[i] / distance;
				}
			}
		}
	}

	for (i = 0; i < numThreads; i++)
		sum += arr_Sum_Threads[i];

	counter = size_cPoints*(size_cPoints - 1);  // num_of_cPoint(num_of_cPoint-1)
	quality = sum / counter;

	free(arr_Sum_Threads);
	free(arr_max_diameter);
	return quality;
}

//q = (d1/D12 + d1/D13 + d2/D21 + d2/D23 + d3/D31 + d3/D32) / 6 

// calculate diameter to each cPoint 
double* calc_cPoint_Diameter(const Point* points, const int size_Points, cPoint* cPointArr, const int size_cPoints)
{
	int i, numThreads;
	double* arr_max_diameter_Threads;
	double* arr_max_diameter = (double*)calloc(size_cPoints, sizeof(double));

	#pragma omp parallel
	{
		int j, cPoint_index, tid = omp_get_thread_num();
		double tmp;

		#pragma omp single
		{
			numThreads = omp_get_num_threads();
			arr_max_diameter_Threads = (double*)calloc(numThreads*size_cPoints, sizeof(double));
		}

		// calculate max diameter for each cPoint
		#pragma omp for
		for (i = 0; i < size_Points; i++)
		{
			cPoint_index = points[i].cPoint_index;
			for (j = i + 1; j < size_Points; j++)
			{
				if (cPoint_index == points[j].cPoint_index)
				{
					tmp = calcPointDistance(points[i].x, points[i].y, points[j].x, points[j].y);
					if (tmp > arr_max_diameter_Threads[(tid*size_cPoints) + cPoint_index])
						arr_max_diameter_Threads[(tid*size_cPoints) + cPoint_index] = tmp;
				}
			}
		}

		#pragma omp for
		for (i = 0; i < size_cPoints; i++)
		{
			for (j = 0; j < numThreads*size_cPoints; j += size_cPoints)
			{
				tmp = arr_max_diameter_Threads[i + j];
				if (tmp > arr_max_diameter[i])
					arr_max_diameter[i] = tmp;
			}
		}
	}
	free(arr_max_diameter_Threads);
	return arr_max_diameter;
}


// change the positions of the points (CUDA)
void change_Points_Positions(Point* arrPoint, const int n, const double dt)
{
	int cuda_blocks;
	if (n > CUDA_THREADS_IN_BLOCK)
		cuda_blocks = n / CUDA_THREADS_IN_BLOCK;
	else
		cuda_blocks = 1;
	
	//if (cuda_blocks > MAX_CUDA_BLOCKS_IN_GRID)
	//	cuda_blocks = MAX_CUDA_BLOCKS_IN_GRID;
	cudaError_t cudaStatus = changePos(arrPoint, n, dt, cuda_blocks, CUDA_THREADS_IN_BLOCK); //  CUDA
																							 
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda failed!\n");
		exit(1);
	}
}

// Copy output points array from GPU buffer to host memory (CUDA)
void copy_Points_To_Source(Point* arrPoint, const int arr_size)
{
	cudaError_t cudaStatus = copyToSource(arrPoint, arr_size); //  CUDA

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda failed!\n");
		exit(1);
	}
}

void write_To_Output_File(const char* fileName, const cPoint* arrCenterCluster, const int size_cPoints, const double time, const double quality)
{
	int i;
	FILE *f;
	errno_t err;
	err = fopen_s(&f, fileName, "w");

	if (err != 0)
	{
		printf("file corrupt or try to enter absolute path to txt file in global variable: filePath");
		exit(1);
	}

	fprintf(f, "First occurrence at t = %.4f  with q = %.4f \nCenters of the clusters: \n", time, quality);
	for (i = 0;i < size_cPoints;i++)
	{
		fprintf(f, "%.4f    %.4f\n", arrCenterCluster[i].x, arrCenterCluster[i].y);
	}
	fclose(f);
}
