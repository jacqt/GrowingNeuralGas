#include "test.h"


vector<double> generateTestData()
{
    //Generates a data point either in the range (0.75,0.75) -> (1,1) or (0,0) -> (.25,.25)
	double g = getRandomDouble(0, 1);
	double lb;
	double ub;
	if (g < 0.5)
	{
		lb = 0.75;
		ub = 1.00;
	}
	else
	{
		lb = 0.00;
		ub = 0.25;
	}
	vector<double> generatedVector = { getRandomDouble(lb, ub), getRandomDouble(lb, ub) }; 
	generatedVector.shrink_to_fit();
	return generatedVector;
}

void runTest()
{
	NeuralGasNetwork* myNet = new NeuralGasNetwork(
        2,
		NODE_CHANGE_RATE,
		NODE_NEIGHBOR_CHANGE_RATE,
		LOCAL_DECREASE_RATE,
		GLOBAL_DECREASE_RATE,
		AGE_MAX,
		TIME_BETWEEN_ADDING_NODES);

	int i = 0;
	while (myNet->nodes.size() < 100 && i < 15000)
	{
		if (i%500 == 0)
            cout << "Iteration " << i << endl;

		myNet->iterate(generateTestData());
		++i;
	}

	LabelGraphNodes(myNet);

    //Now train the perceptron classifer
	PerceptronClassifier* myClassifier = new PerceptronClassifier(myNet,
		2,
		2);

	for (unsigned int i = 0; i != 100; ++i)
	{
		vector<double> testData = generateTestData();
		if (testData[0] > 0.5)
			myClassifier->train_GNG_Perceptron(testData, 1);
		else
			myClassifier->train_GNG_Perceptron(testData, 0);
	}

	while (1)
	{
		double t1;
		double t2;
		cin >> t1 >> t2;
		vector<double> testVec = { t1, t2 };
		int k = myNet->findNearest(testVec);
		cout << "GNG Classifcation: " << myNet->nodes[k]->classification << endl;
		cout << "GNG Error: " << myNet->nodes[k]->getEuclidianDistance(testVec) << endl;
		myClassifier->printOutput(testVec);
		
	}
}
