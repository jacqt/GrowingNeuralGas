#ifndef TEST_H
#define TEST_H

#include "general_include.h"
#include "utils.h"
#include "gng.h"
#include "graph_algorithms.h"
#include "classify_gng.h"

//Generates some test data
vector<double> generateTestData();

//Runs some tests on the test data
void runTest();

#endif