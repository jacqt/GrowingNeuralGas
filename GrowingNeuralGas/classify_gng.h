#ifndef CLASSIFY_GNG_H
#define CLASSIFY_GNG_H

#define LEARNING_RATE       0.1

#include "general_include.h"
#include "gng.h"
#include "utils.h"

//Takes a GNG and creates a classifier from it
class Perceptron;
class PerceptronClassifier;

class Perceptron{
public:
	unsigned int featureSpaceDimension;
	int output;
	vector<double> weights;

	Perceptron(unsigned int newFeatureSpaceDimensions);

	int getOutput(const vector<double> &featureVector);

	void train(const vector<double> &featureVector, int target);

};

class PerceptronClassifier{
public:
	unsigned int featureSpaceDimension;
	vector<Perceptron*> perceptrons;
	NeuralGasNetwork* network;

	PerceptronClassifier(NeuralGasNetwork* newNetwork,
		unsigned int newFeatureSpaceDimensions,
		unsigned int numberOfOutputs);
    
    //Assuming the GNG has already been trained and searhced w/ DFS, now train the classifier
    void train_GNG_Perceptron(const vector<double> &featureVector, int targetNode);

	void printOutput(const vector<double> &featureVector);

};

int indicatorFunction(double i);

//takes a GNG classification and turns it into a feature vector
vector<double> classificationToVector(int classification);



#endif