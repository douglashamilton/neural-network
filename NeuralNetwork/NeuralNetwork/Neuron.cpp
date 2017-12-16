#include "stdafx.h"
#include "Neuron.h"
#include <iostream>

double Neuron::eta = 0.15; // learning rate
double Neuron::alpha = 0.5; // momentum

void Neuron::UpdateInputWeights(Layer &PrevLayer)
{
	// the weights to be updated are in the connection container
	//in the neurons in the preceding layer

	for (unsigned n = 0; n < PrevLayer.size(); ++n)
	{
		Neuron &neuron = PrevLayer[n];
		double oldDeltaWeight = neuron.OutputWeights[MyIndex].DeltaWeight;

		double newDeltaWeight =
			// individual input, magnified by the gradient and the train ratio
			eta
			* neuron.GetOutputValue()
			* Gradient
			// also add momentum = a fraction of the previous delta weight
			+ alpha
			* oldDeltaWeight;

		neuron.OutputWeights[MyIndex].DeltaWeight = newDeltaWeight;
		neuron.OutputWeights[MyIndex].Weight += newDeltaWeight;
	}
}

double Neuron::SumDOW(const Layer &NextLayer) const
{
	double sum = 0.0;

	// sum our contribution of the errors at the nodes we feed
	for (unsigned n = 0; n < NextLayer.size() - 1; ++n)
	{
		sum += OutputWeights[n].Weight * NextLayer[n].Gradient;
	}
	return sum;
}

void Neuron::CalcHiddenGradents(const Layer &NextLayer)
{
	double dow = SumDOW(NextLayer); // sum of derivatives of weights
	Gradient = dow * Neuron::TransferFunctionDerivative(OutputValue);
}

void Neuron::CalcOutputGradients(double TargetVal)
{
	double delta = TargetVal - OutputValue;
	Gradient = delta * Neuron::TransferFunctionDerivative(OutputValue);
}

Neuron::Neuron(unsigned NumberOfOutputs, unsigned n)
{
	for (unsigned c = 0; c < NumberOfOutputs; ++c)
	{
		OutputWeights.push_back(Connection());
		OutputWeights.back().Weight = RandomWeight();
	}
	MyIndex = n;
}

double Neuron::GetOutputValue(void) const {
	return OutputValue;
}

void Neuron::SetOutputValue(const double val)
{
	OutputValue = val;
	return;
}

void Neuron::FeedForward(const Layer &PrevLayer)
{
	double sum = 0.0;
	// sum the previous layer outputs
	// include the bias node from the previous layer

	for (unsigned n = 0; n < PrevLayer.size(); ++n)
	{
		sum += PrevLayer[n].GetOutputValue() *
			PrevLayer[n].OutputWeights[MyIndex].Weight;
	}
	OutputValue = Neuron::TransferFunction(sum);
}

double Neuron::TransferFunction(double x)
{
	// tanh - output 0-1
	return tanh(x);
}

double Neuron::TransferFunctionDerivative(double x)
{
	// tanh derivative
	return 1.0 - x*x;
}

Weights Neuron::GetWeights(void) const
{
	Weights out;
	for (int i = 0; i < OutputWeights.size(); ++i)
	{
		out.push_back(OutputWeights[i].Weight);
		std::cout << OutputWeights[i].Weight << ", ";
	}

	std::cout << std::endl;
	return out;
}