#pragma once
#include <vector>
#include "Neuron.h"
#include <cassert>

typedef std::vector<Weights> LayerWeights;

class Net {
public:
	Net(const std::vector<unsigned> &);
	Net(const unsigned &, const unsigned &);
	void FeedForward(const std::vector<double> &);
	void BackProp(const std::vector<double> &);
	void GetResults(std::vector<double> &) const;
	double GetRecentAverageError(void) const { return RecentAverageError; }
	void GetWeights(std::vector<LayerWeights > &);
	double CalculateAverageError(const std::vector<double> &, const std::vector<double> &);
	
private:
	std::vector<Layer> Layers;
	double Error;
	double RecentAverageError = 0.0;
	static double RecentAverageSmoothingFactor;
	int DetermineNumberOfHiddenLayers(unsigned InputSize, unsigned OutputSize);
	void Initialise(const std::vector<unsigned> &);
	double CalculateAverageError(const Layer &ResultVals, const std::vector<double> &);
};


