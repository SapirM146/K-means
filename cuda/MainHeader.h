#pragma once

struct cPoint
{
	double x;
	double y;
};

struct Point
{
	double x;
	double y;
	double vx;
	double vy;
	int cPoint_index; //  center point of cluster
};

// main
void read_From_File(const char* fileName, Point** arrPoint, int* n, int* k, double* t, double* dt, int* limit, double* qM);
cPoint* k_means(Point* arrPoint, const int n, const int k, const double t, const double dt, const int limit, const double qM, const double initial_time, double* finish_time, double* finish_quality, int* result, int* best_Result_Rank, const int myid, const int numprocs);
void Cluster_Point_Min_Distance(Point* points, const int points_size, cPoint* arrCenterCluster, const int arrCluster_size, int* flag);
float calcPointDistance(const double x1, const double y1, const double x2, const double y2);
void calculateNewAvg(Point* arrPoint, const int sizeArrPoint, cPoint* arrCenterCluster, const int sizeArrCPoint);
double calcQuality(const Point* points, const int size_Points, cPoint* cPoints, const int size_cPoints);
double* calc_cPoint_Diameter(const Point* points, const int size_Points, cPoint* cPointArr, const int size_cPoints);
void change_Points_Positions(Point* arrPoint, int n, double dt);
void copy_Points_To_Source(Point* arrPoint, const int arr_size);
void write_To_Output_File(const char* fileName, const cPoint* arrCenterCluster, const int size_cPoints, const double time, const double quality);

