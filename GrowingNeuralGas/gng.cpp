#include "gng.h"

//////////////////////////////////////////////////////////////////////
///////////////////////////NODE EDGE//////////////////////////////////
//////////////////////////////////////////////////////////////////////

//Construct for a node edge
NodeEdge::NodeEdge(NeuralGasNode* n1, NeuralGasNode* n2, double age_)
{
	node1 = n1;
	node2 = n2;
	age = age_;
}

//Returns the neighboring node of the node passed in as an argument
NeuralGasNode* NodeEdge::neighboringNode(NeuralGasNode* node)
{
	if (node == node1)
		return node2;
    return node1;
}

//Node Edge destructor
NodeEdge::~NodeEdge()
{
	for (auto it = node1->edges.begin(); it != node1->edges.end();  ++it)
	{
		if ((*it) == this)
		{
			node1->edges.erase(it);
			break;
		}
	}
	for (auto it = node2->edges.begin(); it != node2->edges.end();  ++it)
	{
		if ((*it) == this)
		{
			node2->edges.erase(it);
			break;
		}
	}
}

//////////////////////////////////////////////////////////////////////
/////////////////////////NEURAL GAS NODE//////////////////////////////
//////////////////////////////////////////////////////////////////////

//Constructor
NeuralGasNode::NeuralGasNode(const vector<double> &initialRefVector)
{
	for (auto it = initialRefVector.begin(); it != initialRefVector.end(); ++it)
		referenceVector.push_back(*it);

}

//Set error += weightChange
void inline NeuralGasNode::updateError(double weightChange)
{
	error += weightChange;
}

//Returns the squared euclidan distance between the reference vector and the feature vector
//We do not perform a square root to save computation
double NeuralGasNode::getEuclidianDistance(const vector<double> &featureVector)
{
	double euclidDistance = 0;
	for (unsigned int i = 0; i != referenceVector.size(); ++i)
		euclidDistance += pow((referenceVector[i] - featureVector[i]), 2);

	return euclidDistance;
}

//Shifts the refernce vector of a node towards a particular destination by a particular amount
void NeuralGasNode::shiftReferenceVector(const vector<double> &dest, double changeAmount)
{
	for (int i = 0; i != dest.size(); ++i)
	{
		double k = (dest[i] - referenceVector[i]) * changeAmount;
		referenceVector[i] += k;
	}
}

//////////////////////////////////////////////////////////////////////
///////////////////////NEURAL GAS NETWORK/////////////////////////////
//////////////////////////////////////////////////////////////////////

//Initializes two random nodes as well as the starting hyper parameters
NeuralGasNetwork::NeuralGasNetwork(unsigned int featureSpaceDimension,
    double newNodeChangeRate,
    double newNodeNeighborChangeRate,
    double newErrorDecreaseFactorLocal,
    double newErrorDecreaseFactorGlobal,
    unsigned int newAgeMax,
    unsigned int newTimeBetweenAddingNodes)
{
    //Initialize the first two nodes
	NeuralGasNode* newNode1 = new NeuralGasNode(getRandomVector(featureSpaceDimension));
	NeuralGasNode* newNode2 = new NeuralGasNode(getRandomVector(featureSpaceDimension));

	NodeEdge* newEdge = new NodeEdge(newNode1, newNode2, 0);

	nodes.push_back(newNode1);
	newNode1->nodeIndex = nodes.size() - 1;
	newNode1->edges.push_back(newEdge);

	nodes.push_back(newNode2);
	newNode2->nodeIndex = nodes.size() - 1;
	newNode2->edges.push_back(newEdge);

	edges.push_back(newEdge);
    
    //Initialize other variables
	nodeChangeRate = newNodeChangeRate;
	nodeNeighborChangeRate = newNodeNeighborChangeRate;
	ageMax = newAgeMax;
	timeBetweenAddingNodes = newTimeBetweenAddingNodes;
	errorDecreaseFactorLocal = newErrorDecreaseFactorLocal;
	errorDecreaseFactorGlobal = newErrorDecreaseFactorGlobal;
	currentIteration = 1;
}

//Finds two closest nodes to a feature vector in terms of the euclidian distance
void NeuralGasNetwork::findTwoNearest(const vector<double> &featureVector,
	unsigned int &closest,
	unsigned int &nextClosest)
{
	double min1 = 999;
	double min2 = 999;
	unsigned int min1Index = 0;
	unsigned int min2Index = 0;

	for (unsigned int i = 0; i != nodes.size(); ++i)
	{
		double dist = nodes[i]->getEuclidianDistance(featureVector);
		if (dist < min1)
		{
			min2 = min1;
			min2Index = min1Index;
			min1 = dist;
			min1Index = i;
		}
		else if (dist < min2)
		{
			min2 = dist;
			min2Index = i;
		}
	}
	closest = min1Index;
	nextClosest = min2Index;
}

//Finds the cloeset node index to a feature vector in terms of euclidian distances
unsigned int NeuralGasNetwork::findNearest(const vector<double> &featureVector)
{
	double min = 999;
	unsigned int minIndex = 0;

	for (unsigned int i = 0; i != nodes.size(); ++i)
	{
		double dist = nodes[i]->getEuclidianDistance(featureVector);
		if (dist < min)
		{
			min = dist;
			minIndex = i;
		}
	}
    return minIndex;
}

unsigned int NeuralGasNetwork::findNodeWithLargestError()
{
	double maxError = 0;
	unsigned int index = 0;
	for (unsigned int i = 0; i != nodes.size(); ++i)
	{
		if (nodes[i]->error > maxError)
		{
			maxError = nodes[i]->error;
			index = i;
		}
	}
	return index;
}

void NeuralGasNetwork::AddNode()
{
	NeuralGasNode* largestErrorNode = nodes[findNodeWithLargestError()];

	double maxNeighborError = 0;
	NeuralGasNode* largestErrorNeighbor;
	if (largestErrorNode->edges[0]->node1 == largestErrorNode)
		largestErrorNeighbor = largestErrorNode->edges[0]->node2;
	else
		largestErrorNeighbor = largestErrorNode->edges[0]->node1;

	NodeEdge* nodeEdge = largestErrorNode->edges[0];
	for (int i = 0; i != largestErrorNode->edges.size(); ++i)
	{
		NeuralGasNode* otherNode = largestErrorNode->edges[i]->neighboringNode(largestErrorNode);
        if (otherNode->error > maxNeighborError)
        {
            maxNeighborError = otherNode->error;
            largestErrorNeighbor = otherNode;
            nodeEdge = largestErrorNode->edges[i];
        }
	}
    
    //Delete the edge
	deleteEdge(nodeEdge);

    //Create the new node
	vector<double> refVector = averageVector(largestErrorNeighbor->referenceVector,
		largestErrorNode->referenceVector);
	NeuralGasNode* newNode = new NeuralGasNode(refVector);
	nodes.push_back(newNode);

    //Create the new edges
	NodeEdge* newEdge1 = new NodeEdge(largestErrorNode, newNode, 0);
	largestErrorNode->edges.push_back(newEdge1);
	newNode->edges.push_back(newEdge1);
    edges.push_back(newEdge1);

	NodeEdge* newEdge2 = new NodeEdge(largestErrorNeighbor, newNode, 0);
	largestErrorNeighbor->edges.push_back(newEdge2);
	newNode->edges.push_back(newEdge2);
    edges.push_back(newEdge2);

    //Decrease the error of the nodes and set the error of the new node
	largestErrorNode->error *= errorDecreaseFactorLocal;
	largestErrorNeighbor->error *= errorDecreaseFactorLocal;
}

void NeuralGasNetwork::iterate(const vector<double> &featureVector)
{
    //Find two nearest nodes to the feature vector
	unsigned int min1;
	unsigned int min2;
	findTwoNearest(featureVector, min1, min2);

	NeuralGasNode* closestNode = nodes[min1];
	NeuralGasNode* secondClosestNode = nodes[min2];

    //Update the local error
	closestNode->updateError(closestNode->getEuclidianDistance(featureVector));

    //Move the closest node closer
	closestNode->shiftReferenceVector(featureVector, nodeChangeRate);

    //Move all the neighbors of the closestnode
	for (auto it = closestNode->edges.begin(); it != closestNode->edges.end(); ++it)
	{
		if ((*it)->node1 == closestNode)
			(*it)->node2->shiftReferenceVector(featureVector, nodeNeighborChangeRate);
		else
			(*it)->node1->shiftReferenceVector(featureVector, nodeNeighborChangeRate);
	}

    //create or reset the edge between closestNode and secondClosestNode
	bool secondClosestIsNeighbor = false;
	for (auto it = closestNode->edges.begin(); it != closestNode->edges.end(); ++it)
	{
		if ((*it)->node1 == closestNode)
		{
			if ((*it)->node2 == secondClosestNode)
			{
				(*it)->age = 0;
				secondClosestIsNeighbor = true;
				break;
			}
		}
		else
		{
			if ((*it)->node1 == secondClosestNode)
			{
				(*it)->age = 0;
				secondClosestIsNeighbor = true;
				break;
            }
		}
	}

	if (!secondClosestIsNeighbor)
	{
		NodeEdge* newEdge = new NodeEdge(closestNode, secondClosestNode, 0);
        
		closestNode->edges.push_back(newEdge);
		secondClosestNode->edges.push_back(newEdge);
		edges.push_back(newEdge);
	}

    //Delete all the old edges
	unsigned int i = 0;
	while (i < edges.size())
	{
		if (edges[i]->age > ageMax)
			deleteEdge(i);
		else
            ++i;
	}

    //Add a node if neccessary
	if (currentIteration % timeBetweenAddingNodes == 0)
		AddNode();

    //Decrease all the errors
	for (auto it = nodes.begin(); it != nodes.end(); ++it)
		(*it)->error *= errorDecreaseFactorGlobal;

    //Increment all the ages of the edges
	for (auto it = edges.begin(); it != edges.end(); ++it)
		++(*it)->age;

    //Check if there are any nodes that have no edges
	i = 0;
	while (i != nodes.size())
	{
		if (nodes[i]->edges.size() == 0)
			deleteNode(i);
		else
			++i;
	}
    //Increment the iteration counter
	++currentIteration;
}

//Write the network to a file
void NeuralGasNetwork::writeToFile()
{
	/*File Format
    - NUMER_OF_NODES
    - REFERNCE VECTOR
    - REFERNCE VECTOR
    - ...
    - NUMBER_OF_EDGES
    - NODE1 NODE2 AGE
    - NODE1 NODE2 AGE
    - ...  
	*/
    
    //First go through the nodes and assign to each one the updated node index
	for (unsigned int i = 0; i != nodes.size(); ++i)
		nodes[i]->nodeIndex = i;

    //Open the file

}

void NeuralGasNetwork::readFromFile(const string fileName)
{
	/*File Format
    - NUMER_OF_NODES
    - REFERNCE VECTOR
    - REFERNCE VECTOR
    - ...
    - NUMBER_OF_EDGES
    - NODE1 NODE2 AGE
    - NODE1 NODE2 AGE
    - ...  
	*/

}

//Deletes an edge from the network
void NeuralGasNetwork::deleteEdge(unsigned int edgeIndex)
{
    delete edges[edgeIndex];
    edges.erase(edges.begin() + edgeIndex);
}

//Deletes an edge from the network
void NeuralGasNetwork::deleteEdge(NodeEdge* edge)
{
	for (unsigned int i = 0; i != edges.size(); ++i)
	{
		if (edges[i] == edge)
		{
			deleteEdge(i);
			break;
		}
	}
}

//Deletes a node from the network
void NeuralGasNetwork::deleteNode(unsigned int nodeIndex)
{
    nodes.erase(nodes.begin() + nodeIndex);
}