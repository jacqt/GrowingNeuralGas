#ifndef GRAPH_ALGO_H
#define GRAPH_ALGO_H

#include "general_include.h"
#include "gng.h"


//Discovers all the connected componenets and identifies each one as seperate classifications
void LabelGraphNodes(NeuralGasNetwork* network);

//Visits a node
void DFS_Visit(NeuralGasNode* node, int currentClass);

#endif