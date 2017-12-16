#include "stdafx.h"
#include "Net.h"
#include <iostream>

double Net::RecentAverageSmoothingFactor = 100.0; // Number of training samples to average over

int Net::DetermineNumberOfHiddenLayers(unsigned InputSize, unsigned OutputSize)
{
	int HiddenSize;
	HiddenSize = (InputSize + OutputSize) / 2 + 1;
	std::cout << "Hidden layer contains " << HiddenSize << " neurons." << std::endl;
	return HiddenSize;
}

void Net::Initialise(const std::vector<unsigned> &Topology)
{
	unsigned NumberOfLayers = Topology.size();

	for (unsigned i = 0; i < NumberOfLayers; ++i)
	{
		Layers.push_back(Layer());
		// NumberOfOutputs of layer[i] is the num inputs of next layer [i+1]
		// NumberOfOutputs of last layer is 0
		unsigned NumberOfOutputs = i == Topology.size() - 1 ? 0 : Topology[i + 1];

		// we have made a new layer, now fill it with neurons, and 
		// add a bias neuron to the layer
		for (int n = 0; n <= Topology[i]; ++n)
		{
			Layers.back().push_back(Neuron(NumberOfOutputs, n));
		}

		// force the bias neurons output to 1.0
		Layers.back().back().SetOutputValue(-1.0);
	}
}

Net::Net(const std::vector<unsigned> &Topology)
{
	Initialise(Topology);
}

Net::Net(const unsigned &InputSize, const unsigned &OutputSize)
{
	unsigned HiddenSize = DetermineNumberOfHiddenLayers(InputSize, OutputSize);
	Initialise({ InputSize, HiddenSize, OutputSize });
}

void Net::GetResults(std::vector<double> &ResultVals) const
{
	ResultVals.clear();

	for (unsigned n = 0; n < Layers.back().size() - 1; ++n)
	{
		ResultVals.push_back(Layers.back()[n].GetOutputValue());
	}
}

void Net::GetWeights(std::vector<LayerWeights> &NetWeights)
{
	int NumLayers = Layers.size();
	for (int i = 0; i < NumLayers - 1; ++i)
	{
		Layer ThisLayer = Layers[i];
		int NumberOfWeights = Layers[i + 1].size() - 1;
		std::vector<Weights> LayerWeights;

		for (int j = 0; j < ThisLayer.size(); ++j)
		{			
			LayerWeights.push_back(ThisLayer[j].GetWeights());
		}
		NetWeights.push_back(LayerWeights);
	}
	
}

void Net::FeedForward(const std::vector<double> &InputValues)
{
	// check the num of InputValues equals number of neuron except bias
	assert(InputValues.size() == Layers[0].size() - 1);

	// assign (latch) the input values into the input neurons
	for (unsigned i = 0; i < InputValues.size(); ++i)
	{
		Layers[0][i].SetOutputValue(InputValues[i]);
	}

	// forward propagation
	for (unsigned LayerNumber = 1; LayerNumber < Layers.size(); ++LayerNumber)
	{
		Layer &PreviousLayer = Layers[LayerNumber - 1];
		for (unsigned n = 0; n < Layers[LayerNumber].size() - 1; ++n)
		{
			Layers[LayerNumber][n].FeedForward(PreviousLayer);
		}
	}
}

double Net::CalculateAverageError(const std::vector<double> &ResultVals, const std::vector<double> &TargetValues)
{
	// calculate total net error
	Error = 0.0;
	for (unsigned n = 0; n < TargetValues.size() - 1; ++n)
	{
		double delta = TargetValues[n] - ResultVals[n];
		Error += delta * delta;
	}
	Error /= TargetValues.size() - 1; // get average error squared
	Error = sqrt(Error); // RMS of difference between target and output

	return Error;
}

double Net::CalculateAverageError(const Layer &ResultVals, const std::vector<double> &TargetValues)
{

	// calculate total net error
	Error = 0.0;
	for (unsigned n = 0; n < TargetValues.size() - 1; ++n)
	{
		double delta = TargetValues[n] - ResultVals[n].GetOutputValue();
		Error += delta * delta;
	}
	Error /= TargetValues.size() - 1; // get average error squared
	Error = sqrt(Error); // RMS of difference between target and output

	return Error;
}

void Net::BackProp(const std::vector<double> &TargetVals)
{

	Layer &OutputLayer = Layers.back();

	// calculate total net error
	Error = CalculateAverageError(OutputLayer, TargetVals);
	// implement a recent average error measurement
	RecentAverageError =
		(RecentAverageError * RecentAverageSmoothingFactor + Error)
		/ (RecentAverageSmoothingFactor + 1.0);

	//calculate output layer gradients
	for (unsigned n = 0; n < OutputLayer.size() - 1; ++n)
	{
		OutputLayer[n].CalcOutputGradients(TargetVals[n]);
	}

	// calc hidden layer gradients
	for (unsigned LayerNum = Layers.size() - 2; LayerNum > 0; --LayerNum)
	{
		Layer &HiddenLayer = Layers[LayerNum];
		Layer &NextLayer = Layers[LayerNum + 1];

		for (unsigned n = 0; n < HiddenLayer.size(); ++n)
		{
			HiddenLayer[n].CalcHiddenGradents(NextLayer);
		}
	}

	// for all layers from outputs to first hidden layer,
	// update connection weights
	for (unsigned LayerNum = Layers.size() - 1; LayerNum > 0; --LayerNum)
	{
		Layer &layer = Layers[LayerNum];
		Layer &PrevLayer = Layers[LayerNum - 1];
		
		for (unsigned n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].UpdateInputWeights(PrevLayer);
		}
	}

}



