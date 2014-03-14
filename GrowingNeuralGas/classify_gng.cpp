#include "classify_gng.h"

Perceptron::Perceptron(unsigned int newFeatureSpaceDimension)
{
	for (unsigned int i = 0; i != newFeatureSpaceDimension + 1; ++i)
		weights.push_back(getRandomDouble(-0.05, 0.05));

	featureSpaceDimension = newFeatureSpaceDimension;
}

int Perceptron::getOutput(const vector<double> &featureVector)
{
	double result = 0;
	for (unsigned int i = 0; i != featureVector.size(); ++i)
		result += weights[i] * featureVector[i];

	result += weights[featureSpaceDimension]; //the bias weight
	output = indicatorFunction(result);
	return output;
}

void Perceptron::train(const vector<double> &featureVector, int target)
{
	getOutput(featureVector);
	for (int i = 0; i != featureSpaceDimension; ++i)
		weights[i] += LEARNING_RATE * (target - output) * featureVector[i];

	weights[featureSpaceDimension] += LEARNING_RATE * (target - output);
}

int indicatorFunction(double i)
{
	if (i < 0.5)
		return 0;
	return 1;
}

vector<double> classificationToVector(int classification, int dimensions)
{
	vector<double> vec;
	for (unsigned int i = 0; i != dimensions; ++i)
	{
		if (i == classification)
			vec.push_back(1);
		else
			vec.push_back(0);
	}
	return vec;
}

PerceptronClassifier::PerceptronClassifier(NeuralGasNetwork* newNetwork,
	unsigned int newFeatureSpaceDimension,
	unsigned int numberOfOutputs)
{
	network = newNetwork;
	featureSpaceDimension = newFeatureSpaceDimension;

	for (unsigned int i = 0; i != numberOfOutputs; ++i)
	{
		Perceptron* newPerceptron = new Perceptron(newFeatureSpaceDimension);
		perceptrons.push_back(newPerceptron);
	}
}

void PerceptronClassifier::train_GNG_Perceptron(const vector<double> &featureVector,
	int targetNode)
{
	NeuralGasNode* node = network->nodes[network->findNearest(featureVector)];
	int gngClass = node->classification;

	vector<double> gngOutput = classificationToVector(gngClass, network->numberOfClassifications);

	for (unsigned int i = 0; i != perceptrons.size(); ++i)
	{
		if (i == targetNode)
			perceptrons[i]->train(gngOutput, 1);
		else
			perceptrons[i]->train(gngOutput, 0);
	}
}

void PerceptronClassifier::printOutput(const vector<double> &featureVector)
{
	NeuralGasNode* node = network->nodes[network->findNearest(featureVector)];
	int gngClass = node->classification;

	vector<double> gngOutput = classificationToVector(gngClass, network->numberOfClassifications);

	cout << "Output: ";
	for (unsigned int i = 0; i != perceptrons.size(); ++i)
		cout << perceptrons[i]->getOutput(gngOutput) << " ";

	cout << endl;
}