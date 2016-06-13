#pragma once
#include "elm_base.h"

template <typename dataT, bool isColMajor = true>
class oselm : public elm_base<dataT, isColMajor>
{
public:
	using matrixT = typename elm_base<dataT, isColMajor>::matrixT;
	using matrixMapT = typename elm_base<dataT, isColMajor>::matrixMapT;

	explicit oselm(int num_neuron, dataT regularity_const = 0, ostream &os = std::cout)
		: elm_base(num_neuron, regularity_const, os)
	{}	// by default regularity_const is set to zero

	virtual ~oselm() {}

	int update(dataT *xTrain_new, dataT *yTrain_new, int batch_size)
	{
		this->m_os << "--Update on oselm begins.--\n";
		this->tic();
		matrixMapT xTrain = this->wrap_data(xTrain_new, batch_size, this->m_featureLength);
		matrixMapT yTrain = this->wrap_data(yTrain_new, batch_size, this->m_numClass);
		matrixT H, lhs, rhs, sol;
		H = this->compute_H_matrix(xTrain);
		rhs = H * m_P;
		lhs = rhs * H.transpose() + matrixT::Identity(H.rows(), H.rows());
		auto isSolved = solve_eigen(sol, lhs, rhs);
		elm_assert(isSolved);
		m_P = m_P - rhs.transpose() * sol;
		this->m_beta = this->m_beta + m_P * H.transpose() * (yTrain - H * this->m_beta);
		this->toc();
		this->m_os << "--Update finishes.--\n";
		return 0;
	}

	int oselm_init_train(dataT *xTrain, int xRows, int xCols,
		dataT *yTrain, int yRows, int yCols)
	{
		/*elm_assert(xRows >= this->m_numNeuron);
		this->elm_train(xTrain, xRows, xCols, yTrain, yRows, yCols);
		matrixMapT xTrainEigen = this->wrap_data(xTrain, xRows, xCols);
		matrixT H = this->compute_H_matrix(xTrainEigen);
		solve_eigen<matrixT>(m_P, H.transpose() * H + this->m_regConst * matrixT::Identity(this->m_numNeuron, this->m_numNeuron),
			matrixT::Identity(this->m_numNeuron, this->m_numNeuron));
		return 0;*/
		return elm_train(xTrain, xRows, xCols, yTrain, yRows, yCols);
	}

	virtual int elm_train(dataT *xTrainPtr, int xRows, int xCols,
		dataT *yTrainPtr, int yRows, int yCols) override
	{
		this->m_os << "--Training begins.--\n";
		this->tic();
		//elm_assert(xRows >= this->m_numNeuron);
		elm_assert(xRows == yRows);
		this->m_featureLength = xCols;
		this->m_numClass = yCols;
		this->m_weight = matrixT(this->m_numNeuron, this->m_featureLength);
		matrixMapT xTrain = wrap_data(xTrainPtr, xRows, xCols);
		matrixMapT yTrain = wrap_data(yTrainPtr, yRows, yCols);
		random_init(this->m_weight.data(), this->m_weight.size(), this->m_range);
		matrixT H = this->compute_H_matrix(xTrain);
		matrixT lhs = H.transpose() * H + matrixT::Identity(this->m_numNeuron, this->m_numNeuron) * this->m_regConst;
		matrixT rhs = H.transpose() * yTrain;
		auto isSolved = solve_eigen(this->m_beta, lhs, rhs);
		elm_assert(isSolved);
		//matrixT P_lhs = H.transpose() * H;
		matrixT P_rhs = matrixT::Identity(this->m_numNeuron, this->m_numNeuron);
		auto isSolvedP = solve_eigen(m_P, lhs, P_rhs);
		elm_assert(isSolvedP);
		this->toc();
		this->m_os << "--Training is finished.--\n";
		return 0;
	}

	vector<dataT> oselm_test(dataT *xTestPtr, int xRows, int xCols,
		dataT *yTestPtr, int yRows, int yCols,
		dataT threshold = 0)
	{
		return this->elm_test(xTestPtr, xRows, xCols, yTestPtr, yRows, yCols, threshold);
	}


protected:
	matrixT m_P;	// The only matrix that is needed to store.  See paper for details.
};