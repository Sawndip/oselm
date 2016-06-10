#pragma once
#include <opencv2/opencv.hpp>
#include <ctime>
#include <functional>
#include <iostream>
#include <vector>

using cv::Mat;
using cv::RNG;
using std::function;
using std::ostream;
using std::vector;
using std::clock_t;

template<typename dataT>
class elm_base
{
public: 
	typedef cv::Mat matrixT;
	typedef function<void(dataT &, const int *)> functionT;
	//elm_base() {}

	// m_featureLength is the input dimension
	// m_numClass is output dimension
	// This two value controls the size of m_weight and m_beta.
	// The size of m_weight is m_numNeuron x m_featureLength.
	// The size of m_beta is m_numNeuron x m_numClass.
	// This two value is not set (meaning that m_weight nad m_beta is not initialized)
	// until the first training, where m_featureLength and m_numClass
	// is determined by the column of xTrain and yTrain respectively.
	explicit elm_base(int num_neuron, dataT regularity_const, ostream &os = std::cout)
		: m_os(os)	// must be initialized
	{
		m_numNeuron = num_neuron;
		m_regConst = regularity_const;
		m_featureLength = 0;
		m_numClass = 0;
		m_rng = RNG(static_cast<uint64>(time(nullptr)));
		m_dataType = parse_data_type<dataT>();
		m_timer = std::clock();
		m_actFunc = [](dataT &t, const int *pos) -> void { t = std::tanh(t); };
		m_range = 0.5;	// heuristic
	}

	virtual ~elm_base() {}

	virtual int elm_train(const matrixT &xTrain, const matrixT &yTrain)
	{
		m_os << "--Training begins.--\n";
		tic();
		m_featureLength = xTrain.cols;
		m_numClass = yTrain.cols;
		m_weight.create(m_numNeuron, m_featureLength, m_dataType);
		random_init(m_weight, m_range);
		matrixT H = compute_H_matrix(xTrain);
		matrixT lhs = H.t() * H + matrixT::eye(m_numNeuron, m_numNeuron, m_dataType) * m_regConst;
		matrixT rhs = H.t() * yTrain;
		auto isSolved = cv::solve(lhs, rhs, m_beta, cv::DECOMP_CHOLESKY);
		CV_Assert(isSolved);
		toc();
		m_os << "--Training is finished.--\n";
		return 0;
	}
	virtual vector<dataT> elm_test(const matrixT &xTest, const matrixT &yTest, dataT threshold = 0)
	{
		m_os << "--Testing begins.--\n";
		tic();
		CV_Assert(m_featureLength != 0);
		CV_Assert(m_numClass != 0);
		CV_Assert(m_featureLength == xTest.cols);
		CV_Assert(yTest.cols == m_numClass);
		CV_Assert(xTest.rows == yTest.rows);
		matrixT yPredicted = compute_score(xTest);
		CV_Assert(yPredicted.size() == yTest.size());
		dataT accuracy = 0.;
		auto statistics = vector<dataT>();
		if (yTest.cols == 1)	// If a two class problem
		{
			int truePos = 0;
			int falsePos = 0;
			int falseNeg = 0;
			int trueNeg = 0;
			for (auto i = 0; i < yTest.rows; ++i)
			{
				auto predictedPositive = yPredicted.at<dataT>(i, 0) > threshold;
				auto groundTruth = abs(yTest.at<dataT>(i, 0) - 1) < 1e-7;
				if (predictedPositive == true)
				{
					if (groundTruth == true)
						truePos++;
					else
						falsePos++;
				}
				else
				{
					if (groundTruth == true)
						falseNeg++;
					else
						trueNeg++;
				}
			}
			accuracy = static_cast<dataT>(truePos + trueNeg) / (truePos + trueNeg + falsePos + falseNeg);
			auto falseAlarm = static_cast<dataT>(falsePos) / (falsePos + trueNeg);
			auto probClassification = static_cast<dataT>(truePos) / (truePos + falseNeg);
			m_os << "Probability of Detection/Classification: " << probClassification << "\n";
			m_os << "False Alarm: " << falseAlarm << "\n";
			statistics.push_back(accuracy);
			statistics.push_back(probClassification);
			statistics.push_back(falseAlarm);
		}
		else	// multiclass problem
		{
			int trueCount = 0;
			int predictedClass[2];
			int trueClass[2];
			for (int i = 0; i < yTest.rows; ++i)
			{
				minMaxIdx(yPredicted.row(i), NULL, NULL, NULL, predictedClass);
				minMaxIdx(yTest.row(i), NULL, NULL, NULL, trueClass);
				CV_Assert(predictedClass[0] == trueClass[0]);
				CV_Assert(predictedClass[0] == 0);
				if (predictedClass[1] == trueClass[1])
				{
					trueCount++;
				}
			}
			accuracy = static_cast<dataT>(trueCount) / yTest.rows;
			statistics.push_back(accuracy);
		}
		m_os << "Accuracy: " << accuracy << "\n";
		toc();
		m_os << "--Testing is finished.--\n";
		return statistics;
	}
	virtual matrixT compute_score(const matrixT &features)
	{
		CV_Assert(m_featureLength != 0);
		CV_Assert(m_numClass != 0);
		CV_Assert(m_featureLength == features.cols);
		return compute_H_matrix(features) * m_beta;
	}


	// utilities
	void set_seed(uint64 seed) { m_rng = RNG(seed); }
	ostream &get_stream() const { return m_os; }
	void set_act_func(const functionT &func) { m_actFunc = func; }
	void set_random_init_range(dataT r) { m_range = r; }
	// initialize each entry of mat uniformly in [-range, range].
	int random_init(matrixT &mat, dataT range)
	{
		CV_Assert(mat.rows > 0 && mat.cols > 0);
		m_rng.fill(mat, RNG::UNIFORM, -range, range);
		return 0;
	}
	matrixT compute_H_matrix(const matrixT &input_mat)
	{
		CV_Assert(m_featureLength != 0);
		CV_Assert(input_mat.cols == m_featureLength);
		matrixT H = input_mat * m_weight.t();
		H.forEach<dataT>(m_actFunc);
		return H;
	}
	clock_t tic() { m_timer = std::clock(); return m_timer; }
	dataT toc() const
	{
		auto elapsed_seconds = static_cast<dataT>(std::clock() - m_timer) / CLOCKS_PER_SEC;
		m_os << "Elapsed time: " << elapsed_seconds << "s.\n";
		return elapsed_seconds;
	}
	// deduce data type into opencv type
	template<typename someType> static int parse_data_type() { CV_Assert(false); return -1; }
	template<> static int parse_data_type<float>() { return CV_32FC1; }
	template<> static int parse_data_type<double>() { return CV_64FC1; }
	template<> static int parse_data_type<int>() { return CV_32SC1; }

protected:
	matrixT m_weight;
	matrixT m_beta;
	int m_numNeuron;
	int m_featureLength;
	int m_numClass;
	dataT m_regConst;
	RNG m_rng;
	int m_dataType;
	dataT m_range; // see function random_init
	functionT m_actFunc;	// activation function
	ostream &m_os; // stream for logging
	clock_t m_timer; // timing
};