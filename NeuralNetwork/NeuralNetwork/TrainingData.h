#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

class TrainingData {
public:
	TrainingData(const std::string filename);
	bool isEoF(void) { return TrainingDataFile.eof(); };
	void getTopology(std::vector<unsigned> &topology);

	//returns the number of input values read from the file:
	unsigned getNextInputs(std::vector<double> &inputVals);
	unsigned getTargetOutputs(std::vector<double> &targetOutputVals);
	
private:
	std::ifstream TrainingDataFile;
};

void TrainingData::getTopology(std::vector<unsigned> &topology)
{
	std::string line;
	std::string label;

	std::getline(TrainingDataFile, line);
	std::stringstream ss(line);
	ss >> label;
	if (this->isEoF() || label.compare("topology:") != 0)
	{
		abort();
	}

	while (!ss.eof())
	{
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}

	return;
}

TrainingData::TrainingData(const std::string filename)
{
	TrainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(std::vector<double> &inputVals)
{
	inputVals.clear();

	std::string line;
	std::getline(TrainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("in:") == 0) {
		double OneValue;
		while (ss >> OneValue)
		{
			inputVals.push_back(OneValue);
		}
	}
	return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(std::vector<double> &targetOutputValues)
{
	targetOutputValues.clear();

	std::string line;
	std::getline(TrainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("out:") == 0)
	{
		double OneValue;
		while (ss >> OneValue) 
		{
			targetOutputValues.push_back(OneValue);
		}
	}
	return targetOutputValues.size();
}