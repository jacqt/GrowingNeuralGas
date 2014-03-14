#include "graph_algorithms.h"

void LabelGraphNodes(NeuralGasNetwork* network)
{
    //Reset all the classes to class -1
	for (auto nodeIt = network->nodes.begin(); nodeIt != network->nodes.end(); ++nodeIt)
		(*nodeIt)->classification = -1;

    //Perform a DFS on the nodes
	int currentClass = 0;
	for (unsigned int i = 0; i != network->nodes.size(); ++i)
	{
		NeuralGasNode* curNode = network->nodes[i];
		if (curNode->classification == -1) // if we have not visited this node
		{
			DFS_Visit(curNode, currentClass); // visit it
			network->numberOfClassifications = currentClass + 1;
			++currentClass; // we have visited all nodes connected to curNode; increment class counter
		}
	}
}

void DFS_Visit(NeuralGasNode* node, int currentClass)
{
	node->classification = currentClass;
	for (unsigned int i = 0; i != node->edges.size(); ++i)
	{
		NeuralGasNode* neighbor = node->edges[i]->otherNode(node);
		if (neighbor->classification == -1) // if we have not visited this node
			DFS_Visit(neighbor, currentClass); // visit it
	}
}