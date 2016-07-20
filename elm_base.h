#ifndef __ELM_BASE_H__
#define __ELM_BASE_H__


#include <ctime>
#include <functional>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <algorithm>
#include <type_traits>
#include <string>
// #include <experimental/filesystem>
//#include <boost/filesystem.hpp>

using std::function;
using std::ostream;
using std::fstream;
using std::vector;
using std::string;
using std::clock_t;
using std::mt19937;
using std::random_device;
using std::uniform_real_distribution;
using std::generate_n;
using std::copy_n;
using std::transform;
using std::bind;
// namespace fs = std::experimental::filesystem;
//namespace fs = boost::filesystem;

#define elm_assert eigen_assert
template<typename dataT> int random_init(dataT *mat, int size, dataT range);
template<typename eigenMatrixT> bool solve_eigen(eigenMatrixT &sol, const eigenMatrixT &lhs, const eigenMatrixT &rhs);
size_t get_hash(const string &str);
// Serialization for matrix data
template<typename eigenMatrixT> int serialize(const eigenMatrixT &mat, fstream &out, const string &matname,
	typename std::enable_if<std::is_class<eigenMatrixT>::value>::type* = nullptr)	// SFINAE
{
	using dataT = typename Eigen::internal::traits<eigenMatrixT>::Scalar;
	// fstream out(filename, std::ios::out | std::ios::app | std::ios::binary);
	// if (!out.is_open())
	// {
	// 	std::cout << "Cannot open file " << filename << std::endl;
	// 	return 1;
	// }
	elm_assert(out.is_open());
	size_t magic = get_hash(matname);
	auto nrows = (int)mat.rows();
	auto ncols = (int)mat.cols();
	out.write((char *)&magic, sizeof(size_t));
	out.write((char *)&nrows, sizeof(int));
	out.write((char *)&ncols, sizeof(int));
	out.write((char *)(mat.data()), sizeof(dataT)*nrows*ncols);
	// out.close();
	return 0;
}
// Serialization for scalar type
template<typename scalarT> int serialize(scalarT scalar, fstream &out, const string &scalarname,
	typename std::enable_if<std::is_fundamental<scalarT>::value>::type* = nullptr)
{
	// fstream out(filename, std::ios::out | std::ios::app | std::ios::binary);
	// if (!out.is_open())
	// {
	// 	std::cout << "Cannot open file " << filename << std::endl;
	// 	return 1;
	// }
	elm_assert(out.is_open());
	size_t magic = get_hash(scalarname);
	out.write((char *)&magic, sizeof(size_t));
	out.write((char *)&scalar, sizeof(scalarT));
	// out.close();
	return 0;
}
// Deserialization for matrix type
template<typename eigenMatrixT>
int deserialize(eigenMatrixT &m, fstream &in, const string &matname,
	typename std::enable_if<std::is_class<eigenMatrixT>::value>::type* = nullptr)
{
	using dataT = typename Eigen::internal::traits<eigenMatrixT>::Scalar;
	elm_assert(in.is_open());
	size_t magic;
	int nrows, ncols;
	in.read((char *)&magic, sizeof(size_t));
	elm_assert(magic == get_hash(matname));
	in.read((char *)&nrows, sizeof(int));
	in.read((char *)&ncols, sizeof(int));
	m.resize(nrows, ncols);
	in.read((char *)(m.data()), sizeof(dataT)*nrows*ncols);
	return 0;
}
// Deserialization for scalar type
template<typename scalarT>
int deserialize(scalarT &scalar, fstream &in, const string &scalarname,
	typename std::enable_if<std::is_fundamental<scalarT>::value>::type* = nullptr) // SFINAE
{
	elm_assert(in.is_open());
	size_t magic;
	in.read((char *)&magic, sizeof(size_t));
	elm_assert(magic == get_hash(scalarname));
	in.read((char *)&scalar, sizeof(scalarT));
	return 0;
}

template<typename dataT, bool isColMajor = true>
class elm_base
{
	using C = std::integral_constant<int, Eigen::ColMajor>;
	using R = std::integral_constant<int, Eigen::RowMajor>;
	using Cond = std::conditional<isColMajor, C, R>;
public: 
	// Deduce Column or Row major at compile time using std::conditional
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
		matrixMapT xTrain = wrap_data(xTrainPtr, xRows, xCols);
		matrixMapT yTrain = wrap_data(yTrainPtr, yRows, yCols);
		random_init(m_weight.data(), (int)m_weight.size(), m_range);
		matrixT H = compute_H_matrix(xTrain);
		matrixT lhs = H.transpose() * H + matrixT::Identity(m_numNeuron, m_numNeuron) * m_regConst;
		matrixT rhs = H.transpose() * yTrain;
		auto isSolved = solve_eigen(m_beta, lhs, rhs);
		elm_assert(isSolved);
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
		matrixMapT xTest = wrap_data(xTestPtr, xRows, xCols);
		matrixMapT yTest = wrap_data(yTestPtr, yRows, yCols);
		matrixT yPredicted = compute_score(xTest);
		elm_assert(yPredicted.rows() == yTest.rows());
		elm_assert(yPredicted.cols() == yTest.cols());
		dataT accuracy = 0.;
		auto statistics = vector<dataT>();
		if (yTest.cols() == 1)	// If a two class problem, notice there should be some ambingurity
								// with cols == 1 or 2.
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
			elm_assert(truePos+trueNeg+falsePos+falseNeg == yTest.rows());
			accuracy = (dataT)(truePos + trueNeg) / (dataT)(truePos + trueNeg + falsePos + falseNeg);
			auto falseAlarm = (dataT)(falsePos) / (dataT)(falsePos + trueNeg);
			auto probClassification = (dataT)(truePos) / (dataT)(truePos + falseNeg);
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
		return statistics;	// for two class problem, statistics = {prob_det, false_alarm, acc}
							// for multi-class case, statistics = {accuracy}
	}
	virtual matrixT compute_score(const matrixT &features)
	{
		elm_assert(m_featureLength != 0);
		elm_assert(m_numClass != 0);
		elm_assert(m_featureLength == features.cols());
		return compute_H_matrix(features) * m_beta;
	}
	// Overloadding function to return scores in the scores ptr
	// If scores ptr is allocated outside, specify ptr_is_allocated as true.
	// In this case, the user is responsible to ensure that scores has enough space to hold the data,
	// name nrows x m_numClass.
	// If ptr_is_allocated is set to false, the score ptr will point to continues memory containing the data.
	// User is not responsible for releasing the data since it is the internal ptr of a static Eigen matrix.
	// In this case, ensure that the ptr is not used when compute_score is called next time.
	virtual int compute_score(dataT *scores, dataT *features, int nrows, int ncols, bool ptr_is_allocated = true)
	{
		matrixMapT featuresMatrix = wrap_data(features, nrows, ncols);
		if (ptr_is_allocated)
		{
			matrixT scoresMatrix = compute_score(featuresMatrix);
			elm_assert(scoresMatrix.rows() == nrows);
			elm_assert(scoresMatrix.cols() == m_numClass);
			copy_n(scoresMatrix.data(), scoresMatrix.size(), scores);
		}
		else
		{
			static matrixT scoresMatrix = compute_score(featuresMatrix);
			scores = scoresMatrix.data();
		}
		return 0;
	}

	// utilities
	void set_seed(unsigned seed) { m_rng.seed(seed); }
	ostream &get_stream() const { return m_os; }	//! TODO: add c style streaming or modify
	void set_act_func(const functionT &func) { m_actFunc = func; }
	void set_random_init_range(dataT r) { m_range = r; }
	void set_feature_length(int feat_len) { m_featureLength = feat_len; }
	void set_num_classes(int nclasses) { m_numClass = nclasses; }
	int get_feature_length() const { return m_featureLength; }
	int get_num_classes() const { return m_numClass; }
	int get_num_neuron() const { return m_numNeuron; }
	dataT get_random_init_range() const { return m_range; }
	dataT get_regularity_const() const { return m_regConst; }
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
	// wrap the input data into a matrix
	matrixMapT wrap_data(dataT *data_ptr, int nrows, int ncols)
	{
		return matrixMapT(data_ptr, nrows, ncols);
	}
	virtual int snapshot(const string &filename)
	{
		// fs::path p = filename;
		// if (fs::exists(p))
		// {
		// 	m_os << "Warning: " << filename << " already exists.  It will be earsed.\n";
		// 	fs::remove(p);
		// }
		fstream out(filename, std::ios::out | std::ios::binary);
		int i1 = 1*serialize(this->m_weight, out, "weight");
		int i2 = 2*serialize(this->m_beta, out, "beta");
		int i3 = 4*serialize(this->m_numNeuron, out, "numNeuron");
		int i4 = 8*serialize(this->m_featureLength, out, "featureLength");
		int i5 = 16*serialize(this->m_regConst, out, "regConst");
		int i6 = 32*serialize(this->m_range, out, "range");
		int i7 = 64 * serialize(this->m_numClass, out, "numClasses");
		out.close();
		return i1 + i2 + i3 + i4 + i5 + i6 + i7;	// This has no use but only brings trouble to myself.
	}
	virtual int load_snapshot(const string &filename)
	{
		fstream in(filename, std::ios::in | std::ios::binary);
		if (!in.is_open())
		{
			m_os << "Cannot load snapshot " << filename << "\n";
			return 1;
		}
		deserialize(this->m_weight, in, "weight");
		deserialize(this->m_beta, in, "beta");
		deserialize(this->m_numNeuron, in, "numNeuron");
		deserialize(this->m_featureLength, in, "featureLength");
		deserialize(this->m_regConst, in, "regConst");
		deserialize(this->m_range, in, "range");
		deserialize(this->m_numClass, in, "numClasses");
		in.close();
		return 0;
	}
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

	// TODO: a reasonable copy and assign operator (if using Boost::Serialization this seems necessary)
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

// solve least square problem: lhs*sol = rhs
// Note that for ELM most of the time lhs is positive definite,
// so the Cholesky decomposition is applied.
template<typename eigenMatrixT>
bool solve_eigen(eigenMatrixT &sol, const eigenMatrixT &lhs, const eigenMatrixT &rhs)
{
	//sol = lhs.ldlt().solve(rhs);
	sol = lhs.template selfadjointView<Eigen::Upper>().llt().solve(rhs);
	return lhs.ldlt().info() == Eigen::Success;
}
inline size_t get_hash(const string &str)
{
	size_t val = std::hash<string>{}(str);
	//std::cout << str << ": " << val << std::endl;
	return val;
}

#endif // __ELM_BASE_H__