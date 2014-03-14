#ifndef UTILS_H
#define UTILS_H

#include "general_include.h"

double getRandomDouble(double lowerbound, double upperbound);

vector<double> getRandomVector(const unsigned int featureSpaceDimension);

//returns the vector in between the two  vectors passed in
vector<double> averageVector(const vector<double> &v1, const vector <double> &v2);

#endif