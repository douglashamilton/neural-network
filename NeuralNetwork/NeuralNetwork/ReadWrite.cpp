#include "stdafx.h"
#include "ReadWrite.h"
#include <iostream>
#include <algorithm>

namespace Printing
{
	void showVectorVals(std::string label, std::vector<double> &v)
	{
		std::cout << label << " ";
		for (unsigned i = 0; i < v.size(); ++i)
		{
			std::cout << v[i] << " ";
		}
		std::cout << std::endl;
	}
}

namespace DataHandling
{
	bool LoadWINETrainingData(const std::string filename, std::vector<ClassifiedData > &TrainingData)
	{
		std::ifstream infile(filename);
		double a, b, c, d, f, g, h,i,j,k,l,m,n,o,p,q;
		char e;
		std::string name;
		std::vector<std::vector<double> > out;

		while (infile >> a >> e >> b >> e >> c >> e >> d >> e >> f >> e >> g >> e >> h >> e >> i >> e >> j >> e >> k >> e >> l >> e >> m >> e >> n >> e >> o >> e >> p >> e >> q)
		{
			ClassifiedData record;
			record.Data = { a,b,c,d,f,g,h,i,j,k,l,m,n };
			record.Result = { o,p,q };

			TrainingData.push_back(record);
		}
		return true;
	}

	bool LoadIRISTrainingData(const std::string filename, std::vector<ClassifiedData > &TrainingData)
	{
		std::ifstream infile(filename);
		double a, b, c, d, f, g, h;
		char e;
		std::string name;
		std::vector<std::vector<double> > out;

		while (infile >> a >> e >> b >> e >> c >> e >> d >> e >> f >> e >> g >> e >> h)
		{
			ClassifiedData record;
			record.Data = { a,b,c,d };
			record.Result = { f, g, h };

			TrainingData.push_back(record);
		}
		return true;
	}

	void ShuffleTrainingData(std::vector<ClassifiedData> &a)
	{
		std::random_shuffle(a.begin(), a.end());
	}

	void ExtractTestData(std::vector<ClassifiedData> &Training, std::vector<ClassifiedData > &Test)
	{
		int NumTrainingSets = Training.size() / 10;

		for (int i = 0; i < NumTrainingSets; ++i)
		{
			int RandIndex = rand() % NumTrainingSets;

			Test.push_back(Training[RandIndex]);
			Training.erase(Training.begin() + RandIndex);

		}
		
	}

	void NormaliseTrainingData(std::vector<ClassifiedData> &)
	{

	}
}




