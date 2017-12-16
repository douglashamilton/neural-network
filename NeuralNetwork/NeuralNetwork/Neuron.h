#pragma once
#include <vector>
#include "Connection.h"
#include <cstdlib>
#include <cmath>

class Neuron;

typedef std::vector<Neuron> Layer;
typedef std::vector<double> Weights;


class Neuron
{

public:
	Neuron(unsigned NumberOfOutputs, unsigned MyIndex);
	void FeedForward(const Layer &PrevLayer);
	double GetOutputValue(void) const;
	void SetOutputValue(double);
	void CalcOutputGradients(double x);
	void CalcHiddenGradents(const Layer &NextLayer);
	void UpdateInputWeights(Layer &PrevLayer);
	Weights GetWeights(void) const;

private:
	static double TransferFunction(double x);
	static double TransferFunctionDerivative(double x);
	double OutputValue;
	std::vector<Connection> OutputWeights; // each neuron has a weight for each neuron in layer to right
	static double RandomWeight(void) { return rand() / double(RAND_MAX); }
	unsigned MyIndex;
	double Gradient;
	double SumDOW(const Layer &NextLayer) const;
	static double eta; //eta = 0 - slow learner, 0.2 - med learner, 1.0 - reckless learner
	static double alpha; // alpha = 0 - no momentum, 0.5 - moderate momentum
};

