#pragma once
#include "elm_base.h"
#include <opencv2/opencv.hpp>
#include <vector>

using std::vector;

template<typename dataT>
class oselm : public elm_base<dataT>
{
public:
	//typedef typename elm_base<dataT>::matrixT matrixT;
	typedef cv::Mat matrixT;

	explicit oselm(int num_neuron)
		: elm_base(num_neuron, 0)	// we are NOT using regularized ELM
	{}

	explicit oselm(int num_neuron, dataT regularity_const, ostream &os = std::cout)
		: elm_base(num_neuron, regularity_const, os)
	{}

	virtual ~oselm() {}

	int oselm_train(const matrixT &xTrain, const matrixT &yTrain)
	{
		CV_Assert(xTrain.rows >= this->m_numNeuron);
		this->elm_train(xTrain, yTrain);
		matrixT H = this->compute_H_matrix(xTrain);
		m_P = (H.t() * H).inv(cv::DECOMP_CHOLESKY);
		return 0;
	}

	vector<dataT> oselm_test(const matrixT &xTest, const matrixT &yTest, dataT threshold = 0) 
	{
		return this->elm_test(xTest, yTest, threshold);
	}

	int update(const matrixT &xTrain_new, const matrixT &yTrain_new)
	{
		this->m_os << "--Update on oselm begins.--\n";
		this->tic();
		matrixT H = this->compute_H_matrix(xTrain_new);
		matrixT rhs = H * m_P;
		matrixT lhs = rhs * H.t() + cv::Mat::eye(H.rows, H.rows, H.type());
		matrixT sol;
		auto isSolvedUpdated = cv::solve(lhs, rhs, sol, cv::DECOMP_CHOLESKY);
		CV_Assert(isSolvedUpdated);
		m_P = m_P - rhs.t() * sol;
		this->m_beta = this->m_beta + m_P * H.t() * (yTrain_new - H * this->m_beta);
		this->toc();
		this->m_os << "--Update finishes.--\n";
		return 0;
	}
protected:
	matrixT m_P;	// The only matrix that is needed to store.  See paper for details.
};