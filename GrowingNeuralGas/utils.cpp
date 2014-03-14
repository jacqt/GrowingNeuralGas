#include "utils.h"

double getRandomDouble(double lowerbound, double upperbound)
{
	double f = (double) rand() / RAND_MAX;
	f = lowerbound + f * (upperbound - lowerbound);
	return f;
}

vector<double> getRandomVector(const unsigned int featureSpaceDimension)
{
	vector<double> randomVec;
	for (unsigned int i = 0; i != featureSpaceDimension; ++i)
		randomVec.push_back(getRandomDouble(0.0, 1.0));
	return randomVec;
}

vector<double> averageVector(const vector<double> &v1, const vector <double> &v2)
{
	vector<double> newVec;
	for (int i = 0; i != v1.size(); ++i)
		newVec.push_back((v1[i] + v2[i]) / 2.0);
	return newVec;
}