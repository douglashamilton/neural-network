#pragma once
#include <fstream>
#include <vector>
#include <string>
#include "ClassifiedData.h"


namespace Printing
{

	void showVectorVals(std::string label, std::vector<double> &v);

}

namespace DataHandling
{
	bool LoadIRISTrainingData(const std::string filename, std::vector<ClassifiedData > &TrainingData);
	bool LoadWINETrainingData(const std::string filename, std::vector<ClassifiedData > &TrainingData);
	void ShuffleTrainingData(std::vector<ClassifiedData > &);
	void NormaliseTrainingData(std::vector<ClassifiedData> &);

	// randomly remove subset to use as test data
	void ExtractTestData(std::vector<ClassifiedData > &Training, std::vector<ClassifiedData > &Test);
}


