// NeuralNetwork.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <vector>
#include "Net.h"
#include <iostream>
#include <string>
#include "ReadWrite.h"
#include "ClassifiedData.h"

int main()
{

	int Epochs = 40;

	std::vector<ClassifiedData> TrainingData;
	DataHandling::LoadWINETrainingData("WINEnormalised.csv", TrainingData);
	DataHandling::

	// extract test data and hold
	std::vector<ClassifiedData> TestData;
	DataHandling::ExtractTestData(TrainingData, TestData);

	Net MyNet({ TrainingData.front().Data.size() , TrainingData.front().Result.size() });

	for (int i = 0; i < Epochs; ++i)
	{
		// shuffle
		DataHandling::ShuffleTrainingData(TrainingData);

		std::cout << "Epoch: " << i + 1 << std::endl;

		for (unsigned j = 0; j < TrainingData.size(); ++j)
		{
			// feedforward training data
			MyNet.FeedForward(TrainingData[j].Data);

			// extract the output
			std::vector<double> resultVals;
			MyNet.GetResults(resultVals);
			//Printing::showVectorVals("Result: ", resultVals);

			// train the network
			MyNet.BackProp(TrainingData[j].Result);
		}
		
		// report how well the training is working, averaged over recent samples
		std::cout << "Net recent average error: "
			<< MyNet.GetRecentAverageError() << std::endl;

		double MeanError = 0.0;
		//TODO - test the accuracy on the test set, but dont backprop
		for (unsigned k = 0; k < TestData.size(); ++k)
		{
			// feedforward test data
			MyNet.FeedForward(TestData[k].Data);

			// extract the output
			std::vector<double> resultVals;
			MyNet.GetResults(resultVals);

			double Error  = MyNet.CalculateAverageError(resultVals, TestData[k].Result);
			MeanError += Error;

		}
		MeanError /= TestData.size();
		std::cout << "Mean test error: " << MeanError << std::endl;
	}

	// print the weights on each neuron
	// std::vector<LayerWeights> netWeights;
	// std::cout << "Net weights:" << std::endl;
	// MyNet.GetWeights(netWeights);

	std::cout << "Done." << std::endl;

	return 0;
}


/*
int main()
{
	std::vector<unsigned> topology = { 2, 4, 1 }; // {input layer size, hidden layer size, output layer size }

	Net MyNet(topology);

	for (int i = 0; i < 50; ++i)
	{
		std::cout << "Sample: " << i + 1 << std::endl;
		std::vector<double> inputVals, resultVals, targetVals;

		// random training sets for XOR -- two inputs one output
		int n1 = (int)(2.0*rand() / double(RAND_MAX));
		int n2 = (int)(2.0*rand() / double(RAND_MAX));
		int target = n1^n2;

		// pass inputs into the net
		inputVals = { (double)n1 , (double)n2 };
		showVectorVals("Inputs: ", inputVals);
		assert(inputVals.size() == topology.front());
		MyNet.FeedForward(inputVals);

		//collect the net's actual results
		MyNet.GetResults(resultVals);
		showVectorVals("Output: ", resultVals);

		// train the net what the outputs should have been
		targetVals = { (double)target };
		showVectorVals("Target: ", targetVals);
		assert(targetVals.size() == topology.back());
		MyNet.BackProp(targetVals);

		// report how well the training is working, averaged over recent samples
		std::cout << "Net recent average error: "
			<< MyNet.GetRecentAverageError() << std::endl;

	}
	// print the weights on each neuron
	std::vector<LayerWeights> netWeights;
	std::cout << "Net weights:" << std::endl;
	MyNet.GetWeights(netWeights);



	std::cout << "Done." << std::endl;
	
	return 0;
}*/

