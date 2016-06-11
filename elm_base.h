#pragma once
#include <ctime>
#include <functional>
#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <algorithm>
#include <type_traits>

using std::function;
using std::ostream;
using std::vector;
using std::clock_t;
using std::mt19937;
using std::random_device;
using std::uniform_real_distribution;
using std::generate_n;
using std::transform;
using std::bind;

#define elm_assert eigen_assert
template<typename dataT> int random_init(dataT *mat, int size, dataT range);

template<typename dataT, bool isColMajor = true>
class elm_base
{
	using C = std::integral_constant<int, Eigen::ColMajor>;
	using R = std::integral_constant<int, Eigen::RowMajor>;
	using Cond = std::conditional<isColMajor, C, R>;
public: 
	typedef Eigen::Matrix<dataT, Eigen::Dynamic, Eigen::Dynamic, Cond::type::value> matrixT;
	typedef Eigen::Map<matrixT> matrixMapT;
	typedef function<dataT(const dataT &)> functionT;

	// m_featureLength is the input dimension
	// m_numClass is output dimension
	// This two value controls the size of m_weight and m_beta.
	// The size of m_weight is m_numNeuron x m_featureLength.
	// The size of m_beta is m_numNeuron x m_numClass.
	// This two value is not set (meaning that m_weight nad m_beta is not initialized)
	// until the first training, where m_featureLength and m_numClass
	// is determined by the column of xTrain and yTrain respectively.
	explicit elm_base(int num_neuron, dataT regularity_const, ostream &os = std::cout)
		: m_os(os), m_rng(random_device{}())	// must be initialized
	{
		if (!isColMajor) m_os << "Warning: using row major instead of column major.\n";
		m_numNeuron = num_neuron;
		m_regConst = regularity_const;
		m_featureLength = 0;
		m_numClass = 0;
		m_timer = std::clock();
		m_actFunc = [](const dataT &t) -> dataT { return std::tanh(t); };
		m_range = 0.5;	// heuristic
	}

	virtual ~elm_base() {}

	virtual int elm_train(dataT *xTrainPtr, int xRows, int xCols,
		dataT *yTrainPtr, int yRows, int yCols)
	{
		m_os << "--Training begins.--\n";
		tic();
		elm_assert(xRows == yRows);
		m_featureLength = xCols;
		m_numClass = yCols;
		m_weight = matrixT(m_numNeuron, m_featureLength);
		matrixMapT xTrain = warp_data(xTrainPtr, xRows, xCols);
		matrixMapT yTrain = warp_data(yTrainPtr, yRows, yCols);
		random_init(m_weight.data(), m_weight.size(), m_range);
		matrixT H = compute_H_matrix(xTrain);
		matrixT lhs = H.transpose() * H + matrixT::Identity(m_numNeuron, m_numNeuron) * m_regConst;
		matrixT rhs = H.transpose() * yTrain;
		m_beta = lhs.ldlt().solve(rhs);
		elm_assert(lhs.ldlt().info() == Eigen::Success);
		toc();
		m_os << "--Training is finished.--\n";
		return 0;
	}
	virtual vector<dataT> elm_test(dataT *xTestPtr, int xRows, int xCols, 
		dataT *yTestPtr, int yRows, int yCols, 
		dataT threshold = 0)
	{
		m_os << "--Testing begins.--\n";
		tic();
		elm_assert(m_featureLength != 0);
		elm_assert(m_numClass != 0);
		elm_assert(xCols == m_featureLength);
		elm_assert(yCols == m_numClass);
		elm_assert(xRows == yRows);
		matrixMapT xTest = warp_data(xTestPtr, xRows, xCols);
		matrixMapT yTest = warp_data(yTestPtr, yRows, yCols);
		matrixT yPredicted = compute_score(xTest);
		elm_assert(yPredicted.rows() == yTest.rows());
		elm_assert(yPredicted.cols() == yTest.cols());
		dataT accuracy = 0.;
		auto statistics = vector<dataT>();
		if (yTest.cols() == 1)	// If a two class problem
		{
			int truePos = 0;
			int falsePos = 0;
			int falseNeg = 0;
			int trueNeg = 0;
			for (auto i = 0; i < yTest.rows(); ++i)
			{
				auto predictedPositive = yPredicted(i, 0) > threshold;
				auto groundTruth = abs(yTest(i, 0) - 1) < 1e-7;
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
			int dummy;
			int predictedClass;
			int trueClass;
			for (int i = 0; i < yTest.rows(); ++i)
			{
				yPredicted.row(i).maxCoeff(&dummy, &predictedClass);
				elm_assert(dummy == 0);
				yTest.row(i).maxCoeff(&dummy, &trueClass);
				elm_assert(dummy == 0);
				if (predictedClass == trueClass)
				{
					trueCount++;
				}
			}
			accuracy = static_cast<dataT>(trueCount) / yTest.rows();
			statistics.push_back(accuracy);
		}
		m_os << "Accuracy: " << accuracy << "\n";
		toc();
		m_os << "--Testing is finished.--\n";
		return statistics;
	}
	virtual matrixT compute_score(const matrixT &features)
	{
		elm_assert(m_featureLength != 0);
		elm_assert(m_numClass != 0);
		elm_assert(m_featureLength == features.cols());
		return compute_H_matrix(features) * m_beta;
	}


	// utilities
	void set_seed(unsigned seed) { m_rng.seed(seed); }
	ostream &get_stream() const { return m_os; }
	void set_act_func(const functionT &func) { m_actFunc = func; }
	void set_random_init_range(dataT r) { m_range = r; }
	clock_t tic() { m_timer = std::clock(); return m_timer; }
	double toc() const
	{
		auto elapsed_seconds = static_cast<double>(std::clock() - m_timer) / CLOCKS_PER_SEC;
		m_os << "Elapsed time: " << elapsed_seconds << "s.\n";
		return elapsed_seconds;
	}
	matrixT compute_H_matrix(const matrixT &input_mat)
	{
		elm_assert(m_featureLength != 0);
		elm_assert(input_mat.cols() == m_featureLength);
		matrixT H = input_mat * m_weight.transpose();
		transform(H.data(), H.data() + H.size(), H.data(), m_actFunc);
		return H;
	}
	matrixMapT warp_data(dataT *data_ptr, int nrows, int ncols)
	{
		return matrixMapT(data_ptr, nrows, ncols);

	}
	// deduce data type into opencv type
//	template<typename someType> static int parse_data_type() { elm_assert(false); return -1; }
//	template<> static int parse_data_type<float>() { return CV_32FC1; }
//	template<> static int parse_data_type<double>() { return CV_64FC1; }
//	template<> static int parse_data_type<int>() { return CV_32SC1; }

protected:
	matrixT m_weight;
	matrixT m_beta;
	int m_numNeuron;
	int m_featureLength;
	int m_numClass;
	dataT m_regConst;
	mt19937 m_rng;
	dataT m_range; // see function random_init
	functionT m_actFunc;	// activation function
	ostream &m_os; // stream for logging
	clock_t m_timer; // timing
};

// initialize each entry of data uniformly in [-range, range].
template<typename dataT>
int random_init(dataT *mat, int size, dataT range)
{
	static mt19937 rng(random_device{}());
	uniform_real_distribution<dataT> dist(-range, range);
	generate_n(mat, size, bind(dist, rng));
	return 0;
}




